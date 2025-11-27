# placeholder
import argparse
import math
import pickle
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from config import load_config
from models.diffusion import DiffusionRefiner


class StreamingTokenDataset(IterableDataset):
    def __init__(self, texts, tokenizer, max_seq_len):
        super().__init__()
        self.texts = texts
        self.tokenizer = tokenizer
        self.sep_id = tokenizer.sep_token_id
        self.max_seq_len = max_seq_len
        self.token_buffer = []
        self.text_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        while len(self.token_buffer) < self.max_seq_len + 1:
            txt = self.texts[self.text_index]
            ids = self.tokenizer(txt, add_special_tokens=False).input_ids
            self.token_buffer.extend(ids)
            self.token_buffer.append(self.sep_id)
            self.text_index = (self.text_index + 1) % len(self.texts)

        block = self.token_buffer[: self.max_seq_len + 1]
        self.token_buffer = self.token_buffer[self.max_seq_len :]

        x = torch.tensor(block[:-1], dtype=torch.long)
        y = torch.tensor(block[1:], dtype=torch.long)
        return x, y


def create_optimizer(model, peak_lr):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name.lower() for nd in ["bias", "norm", "layernorm", "rmsnorm", "embedding"]):
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=peak_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    return optimizer


def cosine_schedule_builder(warmup_steps, total_steps):
    warmup_steps = max(1, warmup_steps)
    total_steps = max(warmup_steps + 1, total_steps)

    def _lr_lambda(step: int):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1 + math.cos(math.pi * progress))

    return _lr_lambda


def build_dataloader(cfg, tokenizer):
    dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    if cfg.dataset_text_field not in dataset.column_names:
        raise ValueError(f"Column '{cfg.dataset_text_field}' not found in dataset columns {dataset.column_names}")
    all_texts = dataset[cfg.dataset_text_field]

    stream_ds = StreamingTokenDataset(
        texts=all_texts,
        tokenizer=tokenizer,
        max_seq_len=cfg.max_sequence_length,
    )

    train_dataloader = DataLoader(
        stream_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
    )
    return train_dataloader


def train_loop(cfg, args, device):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.pretrained_tokenizer,
        use_fast=True,
        model_max_length=None,
    )
    vocab_size = cfg.vocab_size or tokenizer.vocab_size

    dataloader = build_dataloader(cfg, tokenizer)
    model = DiffusionRefiner(
        vocab_size=vocab_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        max_seq_len=cfg.max_sequence_length,
        num_timesteps=cfg.diffusion_num_timesteps,
        beta_start=cfg.diffusion_beta_start,
        beta_end=cfg.diffusion_beta_end,
    ).to(device)
    print("Total trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    total_tokens = cfg.estimated_tokens_per_epoch
    total_examples_per_epoch = max(1, total_tokens // cfg.max_sequence_length)
    batches_per_epoch = max(1, total_examples_per_epoch // cfg.batch_size)
    effective_batch_size = cfg.effective_batch_size
    total_steps = batches_per_epoch * cfg.num_epochs
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
    save_every = max(1, int(batches_per_epoch * cfg.save_interval_ratio))

    optimizer = create_optimizer(model, cfg.lr)
    scheduler = LambdaLR(optimizer, cosine_schedule_builder(warmup_steps, total_steps))

    log_timesteps = sorted({max(0, int(t)) for t in (cfg.diffusion_log_timesteps or [])})

    all_losses = []
    print("Starting pure diffusion training...")
    model.train()

    for epoch in range(cfg.num_epochs):
        data_iter = iter(dataloader)
        pbar = tqdm(range(batches_per_epoch), total=batches_per_epoch, desc=f"Epoch {epoch + 1}")

        for batch_idx in pbar:
            x, _ = next(data_iter)
            x = x.to(device, non_blocking=True)
            B = x.size(0)

            t = torch.randint(low=0, high=cfg.diffusion_num_timesteps, size=(B,), device=device)
            noise = torch.randn(B, x.size(1), cfg.embed_dim, device=device)
            pred_noise = model(x, timesteps=t, noise=noise)

            loss = torch.nn.functional.mse_loss(pred_noise, noise)

            # Optional logging for specific timesteps
            aux_losses = []
            with torch.no_grad():
                for lt in log_timesteps:
                    if lt >= cfg.diffusion_num_timesteps:
                        continue
                    mask = t == lt
                    if mask.any():
                        aux_loss = torch.nn.functional.mse_loss(pred_noise[mask], noise[mask])
                        aux_losses.append({"t": lt, "loss": aux_loss.item()})

            loss_norm = loss / cfg.gradient_accumulation_steps
            loss_norm.backward()

            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            loss_entry = {"loss": loss.item()}
            if aux_losses:
                loss_entry["aux_losses"] = aux_losses
            all_losses.append(loss_entry)

            desc = f"Epoch {epoch + 1}, Loss: {loss.item():.4f}"
            if aux_losses:
                aux_desc = ", ".join([f"t{rec['t']}:{rec['loss']:.4f}" for rec in aux_losses])
                desc += f", {aux_desc}"
            pbar.set_description(desc)

            if batch_idx % save_every == 0:
                ckpt_name = f"{cfg.checkpoint_prefix}-diffusion-epoch-{epoch + 1}-batch-{batch_idx}.pt"
                torch.save(model.state_dict(), ckpt_name)
                with open("losses.pkl", "wb") as f:
                    pickle.dump(all_losses, f)

    torch.save(model.state_dict(), f"{cfg.checkpoint_prefix}-diffusion-final.pt")
    with open("losses.pkl", "wb") as f:
        pickle.dump(all_losses, f)
    print("Training complete!")


def parse_args():
    parser = argparse.ArgumentParser(description="Pure diffusion training/testing for refiner.")
    parser.add_argument("--cfg", default="diffuser", help="Name of config file without .json.")
    parser.add_argument("--mode", choices=["train"], default="train", help="Run mode (train only for now).")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        train_loop(cfg, args, device)
    else:
        raise ValueError("Only train mode is implemented for diffusion.")


if __name__ == "__main__":
    main()

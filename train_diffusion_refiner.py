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
from torch.nn import functional as F


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
        use_inverse_vocab_head=getattr(cfg, "diffusion_use_inverse_head", False),
    ).to(device)
    print("Total trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    if cfg.resume_checkpoint_path:
        state_dict = torch.load(cfg.resume_checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Resumed model weights from {cfg.resume_checkpoint_path}")

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
            x, y = next(data_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            B = x.size(0)

            t = torch.randint(low=0, high=cfg.diffusion_num_timesteps, size=(B,), device=device)
            noise = torch.randn(B, x.size(1), cfg.embed_dim, device=device)
            pred_noise, logits = model(x, timesteps=t, noise=noise, return_logits=True)

            # CE weighted toward later timesteps (last 2-3 steps dominate)
            ce_per_token = torch.nn.functional.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                reduction="none",
                ignore_index=tokenizer.pad_token_id,
            ).view(B, -1)
            pad_mask = (y != tokenizer.pad_token_id).float()
            distance_from_end = (cfg.diffusion_num_timesteps - 1 - t).view(B, 1).float()
            ce_sample_weight = torch.full_like(distance_from_end, 0)
            ce_sample_weight[distance_from_end <= 0] = 1.0  # emphasize last 3 steps
            weighted = ce_per_token * pad_mask * ce_sample_weight
            denom = (pad_mask * ce_sample_weight).sum().clamp_min(1e-8)
            ce_loss = weighted.sum() / denom
            mse_loss = torch.nn.functional.mse_loss(pred_noise, noise)
            loss = cfg.diffusion_mse_weight * mse_loss + cfg.diffusion_ce_weight * ce_loss

            # Optional logging for specific timesteps
            aux_losses = []
            with torch.no_grad():
                for lt in log_timesteps:
                    if lt >= cfg.diffusion_num_timesteps:
                        continue
                    mask = t == lt
                    if mask.any():
                        aux_ce = torch.nn.functional.cross_entropy(
                            logits[mask].view(-1, vocab_size),
                            y[mask].view(-1),
                            ignore_index=tokenizer.pad_token_id,
                        )
                        aux_mse = torch.nn.functional.mse_loss(pred_noise[mask], noise[mask])
                        aux_losses.append({"t": lt, "ce": aux_ce.item(), "mse": aux_mse.item()})

            loss_norm = loss / cfg.gradient_accumulation_steps
            loss_norm.backward()

            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            loss_entry = {"loss": loss.item()}
            loss_entry["ce_loss"] = ce_loss.item()
            loss_entry["mse_loss"] = mse_loss.item()
            if aux_losses:
                loss_entry["aux_losses"] = aux_losses
            all_losses.append(loss_entry)

            desc = f"Epoch {epoch + 1}, Loss: {loss.item():.4f}"
            if aux_losses:
                aux_desc = ", ".join([f"t{rec['t']}:CE{rec['ce']:.4f}/MSE{rec['mse']:.4f}" for rec in aux_losses])
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


@torch.no_grad()
def diffusion_rollout_logits(model, input_ids, cfg):
    """
    Deterministic (eta=0) reverse diffusion to produce final start-slot logits.
    input_ids: (B, T)
    returns logits shape (B, T, vocab)
    """
    device = input_ids.device
    B, T = input_ids.shape

    # Build clean start base with positional encodings
    seq_len = T * 2
    if seq_len > model.pe.size(0):
        raise ValueError(f"Sequence length {seq_len} exceeds positional embedding limit {model.pe.size(0)}.")
    pos_embeddings = model.pe[:seq_len, :].unsqueeze(0).to(device)
    start_pos = pos_embeddings[:, 1::2, :]
    start_base = model.start_embed.view(1, 1, model.embed_dim).expand(B, T, model.embed_dim) + start_pos

    # Start from pure noise at highest timestep
    x_noisy = start_base + torch.randn_like(start_base) * model.sqrt_one_minus_alphas_cumprod[-1]
    final_logits = None

    for t_step in reversed(range(cfg.diffusion_num_timesteps)):
        t = torch.full((B,), t_step, device=device, dtype=torch.long)
        pred_noise, logits, x0_pred = model(
            input_ids,
            timesteps=t,
            noisy_start_override=x_noisy,
            return_logits=True,
            return_x0=True,
        )
        final_logits = logits
        if t_step > 0:
            sqrt_alpha_prev = model.sqrt_alphas_cumprod[t_step - 1]
            sqrt_one_minus_prev = model.sqrt_one_minus_alphas_cumprod[t_step - 1]
            x_noisy = sqrt_alpha_prev * x0_pred + sqrt_one_minus_prev * pred_noise  # deterministic (eta=0)

    return final_logits


@torch.no_grad()
def generate_completion(model, tokenizer, device, cfg, prompt, max_new_tokens=None, temperature=None):
    max_new_tokens = max_new_tokens or cfg.generation_max_new_tokens
    temperature = temperature or cfg.generation_temperature

    enc = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = enc.input_ids.to(device)

    for _ in range(max_new_tokens):
        if input_ids.size(1) > cfg.max_sequence_length:
            input_ids = input_ids[:, -cfg.max_sequence_length :]

        logits = diffusion_rollout_logits(model, input_ids, cfg)
        # logits correspond to start slots, aligned 1:1 with tokens
        last_logits = logits[:, -1, :]

        probs = torch.softmax(last_logits / temperature, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token_id], dim=1)

        if next_token_id.item() == tokenizer.sep_token_id:
            break

    output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return output_text


def run_test(cfg, args, device):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.pretrained_tokenizer,
        use_fast=True,
        model_max_length=None,
    )
    vocab_size = cfg.vocab_size or tokenizer.vocab_size

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
        use_inverse_vocab_head=getattr(cfg, "diffusion_use_inverse_head", False),
    ).to(device)

    checkpoint_path = args.checkpoint or cfg.resume_checkpoint_path
    if not checkpoint_path:
        raise ValueError("Provide a checkpoint via --checkpoint or resume_checkpoint_path in the config.")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    max_new_tokens = args.max_new_tokens or cfg.generation_max_new_tokens
    temperature = args.temperature or cfg.generation_temperature

    print("Loaded diffusion model. Type a prompt and press Enter (empty line to quit).")
    while True:
        prompt = input("\nPrompt > ")
        if prompt.strip() == "":
            break

        completion = generate_completion(
            model,
            tokenizer,
            device,
            cfg,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        print(f"\nFull output -> {completion}")

    print("\nExiting.")


def parse_args():
    parser = argparse.ArgumentParser(description="Pure diffusion training/testing for refiner.")
    parser.add_argument("--cfg", default="diffuser", help="Name of config file without .json.")
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="Run mode.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path for test mode.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override generation token budget.")
    parser.add_argument("--temperature", type=float, default=None, help="Override sampling temperature.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        train_loop(cfg, args, device)
    else:
        run_test(cfg, args, device)


if __name__ == "__main__":
    main()

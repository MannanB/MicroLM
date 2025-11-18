import argparse
import math
import pickle

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from config import load_config
from models.base import DecoderOnlyTransformer
from models.recursive import RecursiveTransformer


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

            self.text_index += 1
            if self.text_index >= len(self.texts):
                self.text_index = 0

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


def compute_recursive_loss(
    logits_per_recursion,
    input_tokens,
    target_tokens,
    identity_weight,
    prediction_weight,
    loss_gamma,
    pad_token_id=None,
):
    if not logits_per_recursion:
        raise ValueError("logits_per_recursion must contain at least one tensor.")

    device = input_tokens.device
    B, T = input_tokens.shape
    seq_len = T * 2

    interleaved_targets = torch.stack((input_tokens, target_tokens), dim=2).reshape(B, seq_len)
    weights = torch.full((B, seq_len), prediction_weight, device=device, dtype=torch.float32)
    weights[:, ::2] = identity_weight

    if pad_token_id is not None:
        pad_mask = interleaved_targets == pad_token_id
        weights = weights.masked_fill(pad_mask, 0.0)

    denom = weights.sum().clamp_min(1e-8)
    flat_targets = interleaved_targets.view(-1)
    flat_weights = weights.view(-1)

    total_loss = torch.tensor(0.0, device=device)
    total_weight = 0.0

    for step, logits in enumerate(logits_per_recursion):
        flat_logits = logits.view(B * seq_len, -1)
        per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        weighted = (per_token * flat_weights).sum() / denom
        weight = loss_gamma ** step
        total_loss = total_loss + weight * weighted
        total_weight += weight

    if total_weight == 0.0:
        raise ValueError("Total loss weight is zero. Check recursive loss weights.")

    return total_loss / total_weight


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


def build_model(cfg, vocab_size, device):
    if cfg.model_type.lower() == "recursive":
        model = RecursiveTransformer(
            vocab_size,
            cfg.embed_dim,
            cfg.num_heads,
            cfg.hidden_dim,
            cfg.num_layers,
            cfg.max_sequence_length,
            n_recursions=cfg.recursive_num_recursions,
        ).to(device)
    else:
        model = DecoderOnlyTransformer(
            vocab_size,
            cfg.embed_dim,
            cfg.num_heads,
            cfg.hidden_dim,
            cfg.num_layers,
            cfg.max_sequence_length,
            use_flash_attn=cfg.use_flash_attn,
            flash_attn_dropout=cfg.flash_attn_dropout,
        ).to(device)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="MicroLM training loop.")
    parser.add_argument("--cfg", default="microlm", help="Name of config file without .json.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.cfg)
    model_type = cfg.model_type.lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.pretrained_tokenizer,
        use_fast=True,
        model_max_length=None,
    )
    vocab_size = cfg.vocab_size or tokenizer.vocab_size

    dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    if cfg.dataset_text_field not in dataset.column_names:
        raise ValueError(f"Column '{cfg.dataset_text_field}' not found in dataset columns {dataset.column_names}")
    all_texts = dataset[cfg.dataset_text_field]

    total_tokens = cfg.estimated_tokens_per_epoch
    total_examples_per_epoch = max(1, total_tokens // cfg.max_sequence_length)
    batches_per_epoch = max(1, total_examples_per_epoch // cfg.batch_size)

    effective_batch_size = cfg.effective_batch_size
    total_steps = batches_per_epoch * cfg.num_epochs
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
    save_every = max(1, int(batches_per_epoch * cfg.save_interval_ratio))

    print(f"Total tokens per epoch:   {total_tokens}")
    print(f"Total examples per epoch: {total_examples_per_epoch}")
    print(f"Total batches per epoch:  {batches_per_epoch} (batch_size={cfg.batch_size}, eff_batch_size={effective_batch_size})")
    print(f"Total tokens overall:     {total_tokens * cfg.num_epochs}")
    print(f"Peak LR:                  {cfg.lr}")
    print(f"Scheduler Warm Up Ratio   {cfg.warmup_ratio * 100}% of steps\n")

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

    model = build_model(cfg, vocab_size, device)
    print("Total trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = create_optimizer(model, cfg.lr)
    scheduler = LambdaLR(optimizer, cosine_schedule_builder(warmup_steps, total_steps))

    all_losses = []

    print("Starting training...")
    model.train()

    for epoch in range(cfg.num_epochs):
        data_iter = iter(train_dataloader)
        pbar = tqdm(range(batches_per_epoch), total=batches_per_epoch, desc=f"Epoch {epoch + 1}")

        for batch_idx in pbar:
            x, y = next(data_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if model_type == "recursive":
                logits = model(x, return_all_logits=True)
                loss = compute_recursive_loss(
                    logits,
                    x,
                    y,
                    cfg.recursive_identity_loss_weight,
                    cfg.recursive_prediction_loss_weight,
                    cfg.recursive_loss_gamma,
                    pad_token_id=tokenizer.pad_token_id,
                )
            else:
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size),
                    y.view(-1),
                    ignore_index=tokenizer.pad_token_id,
                )

            loss_norm = loss / cfg.gradient_accumulation_steps
            loss_norm.backward()

            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            all_losses.append(loss.item())
            pbar.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

            if batch_idx % save_every == 0:
                ckpt_name = f"{cfg.checkpoint_prefix}-epoch-{epoch + 1}-batch-{batch_idx}.pt"
                torch.save(model.state_dict(), ckpt_name)
                with open("losses.pkl", "wb") as f:
                    pickle.dump(all_losses, f)

    torch.save(model.state_dict(), f"{cfg.checkpoint_prefix}-final.pt")
    with open("losses.pkl", "wb") as f:
        pickle.dump(all_losses, f)

    print("Training complete!")


if __name__ == "__main__":
    main()

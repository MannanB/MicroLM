import argparse
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer

from config import load_config
from models.base import DecoderOnlyTransformer
from models.recursive import RecursiveTransformer
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


def build_model(cfg, vocab_size, device):
    mtype = cfg.model_type.lower()
    if mtype == "recursive":
        model = RecursiveTransformer(
            vocab_size=vocab_size,
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            max_seq_len=cfg.max_sequence_length,
            n_recursions=cfg.recursive_num_recursions,
            return_all_logits=False,
        ).to(device)
    elif mtype == "diffusion":
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
    else:
        model = DecoderOnlyTransformer(
            vocab_size=vocab_size,
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            max_seq_len=cfg.max_sequence_length,
            use_flash_attn=cfg.use_flash_attn,
            flash_attn_dropout=cfg.flash_attn_dropout,
        ).to(device)
    return model


def evaluate_model(cfg_name, ckpt_path, max_batches, device):
    cfg = load_config(cfg_name)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.pretrained_tokenizer,
        use_fast=True,
        model_max_length=None,
    )
    vocab_size = cfg.vocab_size or tokenizer.vocab_size

    dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    if cfg.dataset_text_field not in dataset.column_names:
        raise ValueError(f"Column '{cfg.dataset_text_field}' not found in dataset columns {dataset.column_names}")
    texts = dataset[cfg.dataset_text_field]

    ds = StreamingTokenDataset(texts, tokenizer, cfg.max_sequence_length)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=True)

    model = build_model(cfg, vocab_size, device)
    effective_ckpt = ckpt_path or cfg.resume_checkpoint_path
    if effective_ckpt:
        state_dict = torch.load(effective_ckpt, map_location=device)
        model.load_state_dict(state_dict)
    model.eval()

    mtype = cfg.model_type.lower()
    total_loss = 0.0
    total_tokens = 0
    total_mse = 0.0
    total_positions = 0
    batches_run = 0

    with torch.no_grad():
        for x, y in loader:
            if batches_run >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)

            if mtype == "diffusion":
                B, T = x.shape
                # Deterministic DDIM-style rollout from t=T-1 to 0
                start_base = model.start_embed.view(1, 1, model.embed_dim).expand(B, T, model.embed_dim) + model.pe[: T * 2, :].unsqueeze(0)[:, 1::2, :]
                x_noisy = start_base + torch.randn_like(start_base) * model.sqrt_one_minus_alphas_cumprod[-1]
                final_logits = None
                for t_step in reversed(range(cfg.diffusion_num_timesteps)):
                    t = torch.full((B,), t_step, device=device, dtype=torch.long)
                    pred_noise, logits, x0_pred = model(
                        x,
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

                if final_logits is None:
                    raise RuntimeError("Diffusion rollout failed to produce logits.")
                loss = F.cross_entropy(
                    final_logits.view(-1, vocab_size),
                    y.view(-1),
                    reduction="sum",
                    ignore_index=tokenizer.pad_token_id,
                )
                total_loss += loss.item()
                total_tokens += (y != tokenizer.pad_token_id).sum().item()
            elif mtype == "recursive":
                logits = model(x)
                # Recursive model returns interleaved logits (tokens, starts,...).
                if logits.dim() != 3:
                    raise ValueError(f"Unexpected logits shape from recursive model: {logits.shape}")
                if logits.size(1) == y.size(1) * 2:
                    logits = logits[:, ::2, :]
                elif logits.size(1) != y.size(1):
                    raise ValueError(f"Recursive logits length {logits.size(1)} incompatible with targets {y.size(1)}")
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size),
                    y.view(-1),
                    reduction="sum",
                    ignore_index=tokenizer.pad_token_id,
                )
                total_loss += loss.item()
                total_tokens += (y != tokenizer.pad_token_id).sum().item()
            else:
                logits = model(x)
                if logits.dim() != 3:
                    raise ValueError(f"Unexpected logits shape from decoder model: {logits.shape}")
                if logits.size(1) != y.size(1):
                    raise ValueError(f"Decoder logits length {logits.size(1)} incompatible with targets {y.size(1)}")
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size),
                    y.view(-1),
                    reduction="sum",
                    ignore_index=tokenizer.pad_token_id,
                )
                total_loss += loss.item()
                total_tokens += (y != tokenizer.pad_token_id).sum().item()

            batches_run += 1

    if mtype == "diffusion":
        avg_mse = total_mse / max(1, total_positions)
        return {"type": mtype, "mse": avg_mse, "positions": total_positions, "batches": batches_run}
    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss) if total_tokens > 0 else float("inf")
    return {"type": mtype, "loss": avg_loss, "ppl": ppl, "tokens": total_tokens, "batches": batches_run}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare multiple models on token-level cross entropy.")
    parser.add_argument("--cfgs", required=True, help="Comma-separated config names (without .json).")
    parser.add_argument("--checkpoints", default="", help="Comma-separated checkpoint paths aligned with cfgs (optional).")
    parser.add_argument("--max-batches", type=int, default=50, help="Number of batches to evaluate per model.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_list = [c.strip() for c in args.cfgs.split(",") if c.strip()]
    ckpt_list = [c.strip() for c in args.checkpoints.split(",")] if args.checkpoints else []
    while len(ckpt_list) < len(cfg_list):
        ckpt_list.append(None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for cfg_name, ckpt in zip(cfg_list, ckpt_list):
        result = evaluate_model(cfg_name, ckpt, args.max_batches, device)
        if result["type"] == "diffusion":
            print(f"{cfg_name}: mse={result['mse']:.6f}, positions={result['positions']}, batches={result['batches']}, ckpt={ckpt or 'none'}")
        else:
            print(f"{cfg_name}: loss={result['loss']:.4f}, ppl={result['ppl']:.2f}, tokens={result['tokens']}, batches={result['batches']}, ckpt={ckpt or 'none'}")


if __name__ == "__main__":
    main()

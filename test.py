import argparse
import torch
from transformers import AutoTokenizer

from config import load_config
from models.base import DecoderOnlyTransformer


def build_model(cfg, vocab_size, device):
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


def load_tokenizer(cfg):
    return AutoTokenizer.from_pretrained(
        cfg.pretrained_tokenizer,
        use_fast=True,
        model_max_length=None,
    )


def load_checkpoint(model, checkpoint_path, device):
    if not checkpoint_path:
        raise ValueError("Provide a checkpoint via --checkpoint or resume_checkpoint_path in the config.")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


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

        logits = model(input_ids)
        last_logits = logits[:, -1, :]

        probs = torch.softmax(last_logits / temperature, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token_id], dim=1)

        if next_token_id.item() == tokenizer.sep_token_id:
            break

    output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return output_text


def interactive_loop(model, tokenizer, device, cfg, args):
    max_new_tokens = args.max_new_tokens or cfg.generation_max_new_tokens
    temperature = args.temperature or cfg.generation_temperature

    print("Loaded model. Type a prompt and press Enter (empty line to quit).")
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
    parser = argparse.ArgumentParser(description="MicroLM inference helper.")
    parser.add_argument("--cfg", default="microlm", help="Name of the config file (without .json).")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path to load. Overrides resume_checkpoint_path from the config.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override generation token budget.")
    parser.add_argument("--temperature", type=float, default=None, help="Override sampling temperature.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(cfg)
    vocab_size = cfg.vocab_size or tokenizer.vocab_size

    model = build_model(cfg, vocab_size, device)
    checkpoint_path = args.checkpoint or cfg.resume_checkpoint_path
    load_checkpoint(model, checkpoint_path, device)

    interactive_loop(model, tokenizer, device, cfg, args)


if __name__ == "__main__":
    main()

from pydantic import BaseModel
from typing import Optional
import json


class LMConfig(BaseModel):
    embed_dim: int
    num_heads: int
    hidden_dim: int
    num_layers: int
    max_sequence_length: int

    pretrained_tokenizer: str
    vocab_size: Optional[int] = None

    gradient_accumulation_steps: int
    batch_size: int
    num_epochs: int

    warmup_ratio: float
    lr: float

    dataset_name: str
    dataset_split: str = "train"
    dataset_text_field: str = "text"
    estimated_tokens_per_epoch: int

    checkpoint_prefix: str
    resume_checkpoint_path: Optional[str] = None

    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = True
    save_interval_ratio: float = 0.1

    effective_batch_size: int

    use_flash_attn: bool = False
    flash_attn_dropout: float = 0.0

    generation_max_new_tokens: int = 500
    generation_temperature: float = 1.0


def load_config(name: str) -> LMConfig:
    with open(f"./configs/{name}.json", "r", encoding="utf-8") as f:
        json_data = json.loads(f.read())

    assert json_data["embed_dim"] % json_data["num_heads"] == 0

    json_data["effective_batch_size"] = (
        json_data["gradient_accumulation_steps"] * json_data["batch_size"]
    )

    if isinstance(json_data["lr"], str) and json_data["lr"].lower() == "auto":
        json_data["lr"] = 3e-4 * (
            json_data["effective_batch_size"] * json_data["max_sequence_length"] / 256_000
        ) ** 0.5

    return LMConfig.model_validate(json_data)

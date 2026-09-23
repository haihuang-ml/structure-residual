from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TrainConfig:
    data_dir: str = "./data/fineweb_edu_gpt2"
    output_dir: str = "./runs/smoke_tiny"
    seed: int = 82
    seq_len: int = 1024
    target_tokens: int = 100_000_000
    eval_tokens: int = 1_000_000
    eval_tokens_final: int | None = None
    eval_interval_tokens: int = 10_000_000
    save_interval_tokens: int = 0
    log_interval: int = 10

    model_size: str = "gpt2-small"
    n_layer: int | None = None
    n_head: int | None = None
    n_embd: int | None = None
    n_positions: int | None = None
    dropout: float = 0.0
    init_from_pretrained: bool = False

    bottleneck_mode: str | None = None
    bottleneck_concept_k: int = 8
    bottleneck_layers: str = "all"

    continuous_concept_mode: str = "none"
    continuous_concept_lambda: float = 0.01
    continuous_concept_layer: str = "mid"
    continuous_concept_dropout: float | None = None
    continuous_concept_adaptive_dropout: float | None = None
    continuous_concept_adaptive_dropout_history: int = 32

    learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    lr_scheduler: str = "cosine"
    lr_decay_tokens: int | None = None
    warmup_steps: int = 0
    warmup_tokens: int = 5_000_000
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    batch_size: int = 8
    grad_accum_steps: int = 8
    num_workers: int = 4
    bf16: bool = True
    fp16: bool = False
    torch_compile: bool = False

    wandb: bool = True
    wandb_project: str = "structure-residual"
    wandb_run_name: str | None = None
    wandb_group: str | None = None
    resume_from: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MODEL_SIZES: dict[str, dict[str, int]] = {
    "tiny": {"n_layer": 4, "n_head": 4, "n_embd": 256},
    "gpt2-small": {"n_layer": 12, "n_head": 12, "n_embd": 768},
    "gpt2-medium": {"n_layer": 24, "n_head": 16, "n_embd": 1024},
}


def str_to_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def model_dims(config: TrainConfig) -> dict[str, int]:
    if config.model_size == "custom":
        if config.n_layer is None or config.n_head is None or config.n_embd is None:
            raise ValueError("custom model_size requires --n_layer, --n_head, and --n_embd")
        dims = {
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
        }
    else:
        if config.model_size not in MODEL_SIZES:
            known = ", ".join(sorted([*MODEL_SIZES, "custom"]))
            raise ValueError(f"Unknown model_size {config.model_size!r}. Expected one of: {known}")
        dims = dict(MODEL_SIZES[config.model_size])

    if config.n_layer is not None:
        dims["n_layer"] = config.n_layer
    if config.n_head is not None:
        dims["n_head"] = config.n_head
    if config.n_embd is not None:
        dims["n_embd"] = config.n_embd
    return dims

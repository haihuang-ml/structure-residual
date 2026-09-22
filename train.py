from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from config import TrainConfig, str_to_bool
from dataset import PackedTokenDataset
from model import build_model, select_continuous_concept_hidden_state


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size <= 1:
        return value

    value = value.clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)

    return value / world_size


def reduce_sum(value, world_size: int, device: torch.device) -> torch.Tensor:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return tensor


def normalized_bottleneck_mode(config: TrainConfig) -> str | None:
    mode = config.bottleneck_mode
    if mode is None:
        return None
    lowered = mode.lower()
    if lowered in {"", "none", "null", "off", "false"}:
        return None
    return lowered


def default_wandb_run_name(config: TrainConfig) -> str:
    model_names = {
        "gpt2-small": "g2s",
        "gpt2-medium-ish": "g2m",
        "tiny": "tiny",
        "custom": "custom",
    }
    bottleneck_mode = normalized_bottleneck_mode(config)
    mode_names = {
        None: "v",
        "boundary": "bbc",
        "sliding_window": "bsw",
        "random_boundary": "brb",
        "random_sparse": "brs",
    }
    mode = mode_names.get(bottleneck_mode, str(bottleneck_mode))

    if config.seq_len % 1024 == 0:
        seq_name = f"s{config.seq_len // 1024}k"
    else:
        seq_name = f"s{config.seq_len}"
    lr_name = f"lr{config.learning_rate / 1e-4:g}"

    parts = [
        mode,
        model_names.get(config.model_size, config.model_size),
        seq_name,
        lr_name,
    ]
    if not config.bf16 and not config.fp16:
        parts.append("fp32")
    if bottleneck_mode is not None:
        parts.append(f"k{config.bottleneck_concept_k}")
        if config.bottleneck_layers != "all":
            bottleneck_layers = {"mid": "m", "last": "f"}.get(config.bottleneck_layers, config.bottleneck_layers)
            parts.append(f"bl{bottleneck_layers}")

    continuous_concept_mode = normalized_continuous_concept_mode(config)
    if continuous_concept_mode != "none":
        layer_name = {"mid": "m", "middle": "m", "final": "f", "last": "f"}.get(config.continuous_concept_layer,
                                                                                config.continuous_concept_layer)
        mode_name = {"cosine": "c", "cosine_delta": "d", "norm": "n", "norm_l2": "nl2", "l2": "l2", "mse": "m",
                     "mse_r": "mr", "mse_rs": "mrs", "mse_rsr": "mrsr", "mse_nm": "mnm", "mse_nmp": "mnmp",
                     "mse2": "m2", "mse_sg": "msg", "mse_rsg": "mrsg"}.get(
            continuous_concept_mode,
            continuous_concept_mode,
        )
        parts.append(f"c{mode_name}{layer_name}l{config.continuous_concept_lambda:g}")
        if config.continuous_concept_dropout is not None:
            parts.append(f"ccd{config.continuous_concept_dropout:g}")
        if config.continuous_concept_adaptive_dropout is not None:
            parts.append(
                f"ccad{config.continuous_concept_adaptive_dropout:g}"
                f"h{config.continuous_concept_adaptive_dropout_history}"
            )
    parts.append(f"s{config.seed}")
    return "-".join(parts)


def setup_wandb(config: TrainConfig, rank: int):
    if not config.wandb or not is_main(rank):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Install wandb or run with --wandb false") from exc

    return wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name or default_wandb_run_name(config),
        group=config.wandb_group,
        config=config.to_dict(),
        dir=config.output_dir,
    )


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train GPT-2-style language models on packed token data.")
    defaults = TrainConfig()
    bool_fields = {f.name for f in fields(TrainConfig) if f.type in {bool, "bool"}}
    optional_int_fields = {
        "n_layer",
        "n_head",
        "n_embd",
        "n_positions",
        "lr_decay_tokens",
        "eval_tokens_final",
    }
    optional_float_fields = {
        "continuous_concept_lambda_sg",
        "continuous_concept_dropout",
        "continuous_concept_adaptive_dropout",
    }
    for f in fields(TrainConfig):
        default = getattr(defaults, f.name)
        if f.name in bool_fields:
            arg_type = str_to_bool
        elif f.name in optional_int_fields:
            arg_type = int
        elif f.name in optional_float_fields:
            arg_type = float
        else:
            arg_type = type(default) if default is not None else str
        parser.add_argument(f"--{f.name}", default=default, type=arg_type)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def setup_distributed() -> tuple[bool, int, int, int]:
    if "RANK" not in os.environ:
        return False, 0, 1, 0
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def cleanup_distributed(enabled: bool) -> None:
    if enabled:
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def verify_bottleneck_attention_startup(model, config: TrainConfig, rank: int) -> None:
    if normalized_bottleneck_mode(config) is None:
        return

    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    transformer = raw_model.transformer
    attn_implementation = getattr(transformer.config, "_attn_implementation", None)
    if attn_implementation != "eager":
        raise RuntimeError(f"BBC/bottleneck attention requires eager attention, got {attn_implementation!r}")

    attention_allowed_density = float(getattr(raw_model, "attention_allowed_density", 0.0))
    if attention_allowed_density <= 0.0:
        raise RuntimeError("BBC/bottleneck attention mask was not patched; attention_allowed_density is zero")

    if is_main(rank):
        print(f"attn_implementation = {attn_implementation}")
        print(f"attention_allowed_density = {attention_allowed_density:.3f}")
        print(f"bottleneck_layers = {config.bottleneck_layers}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_meta(data_dir: Path) -> dict[str, Any]:
    with (data_dir / "meta.json").open() as f:
        return json.load(f)


def make_loader(
        path: Path,
        seq_len: int,
        dtype: str,
        batch_size: int,
        num_workers: int,
        random_sampling: bool,
        seed: int,
        distributed: bool,
) -> DataLoader:
    dataset = PackedTokenDataset(
        str(path),
        seq_len=seq_len,
        random_sampling=False,
        seed=seed,
        dtype=dtype,
    )
    sampler = DistributedSampler(dataset, shuffle=random_sampling) if distributed else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=random_sampling if sampler is None else False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def output_value(outputs, name: str):
    if isinstance(outputs, dict):
        return outputs[name]
    return getattr(outputs, name)


def normalized_continuous_concept_mode(config: TrainConfig) -> str:
    mode = config.continuous_concept_mode
    if mode is None:
        return "none"
    lowered = mode.lower()
    aliases = {
        "": "none",
        "none": "none",
        "null": "none",
        "off": "none",
        "false": "none",
        "cos": "cosine",
        "cosine": "cosine",
        "delta": "cosine_delta",
        "cosine_delta": "cosine_delta",
        "norm": "norm",
        "norm_l2": "norm_l2",
        "l2": "l2",
        "mse": "mse",
        "mse_r": "mse_r",
        "mse_rs": "mse_rs",
        "mse_rsr": "mse_rsr",
        "mse_nm": "mse_nm",
        "mse_nmp": "mse_nmp",
        "mse2": "mse2",
        "mse_sg": "mse_sg",
        "mse_rsg": "mse_rsg",
    }
    return aliases.get(lowered, lowered)


def continuous_concept_lambda_at_tokens(config: TrainConfig, trained_tokens: int) -> float:
    del trained_tokens
    if normalized_continuous_concept_mode(config) == "none":
        return 0.0
    return config.continuous_concept_lambda


def set_continuous_concept_lambda(model, value: float) -> None:
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    if hasattr(raw_model, "continuous_concept_lambda"):
        raw_model.continuous_concept_lambda = value


def hidden_layer_eval_sums(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    diffs = hidden[:, 1:, :] - hidden[:, :-1, :]
    step_mse = diffs.pow(2).mean()
    step_count = 1
    cosine_sum = torch.cosine_similarity(hidden[:, :-1, :], hidden[:, 1:, :], dim=-1).sum()
    cosine_count = hidden.shape[0] * max(0, hidden.shape[1] - 1)
    return step_mse, cosine_sum, step_count, cosine_count


def continuous_concept_eval_sums(hidden: torch.Tensor) -> tuple[
    torch.Tensor, int, torch.Tensor, int, torch.Tensor, int]:
    """Return sums and element/pair counts for configured-layer CC diagnostics."""
    l2_sum = hidden.pow(2).sum()
    l2_count = hidden.numel()
    if hidden.shape[1] < 2:
        empty = torch.zeros((), dtype=hidden.dtype, device=hidden.device)
        return l2_sum, l2_count, empty, 0, empty, 0
    previous = hidden[:, :-1, :]
    current = hidden[:, 1:, :]
    dot_sum = (current * previous).sum()
    dot_count = current.numel()
    cos_distance_sum = (1.0 - torch.cosine_similarity(current, previous, dim=-1)).sum()
    cos_distance_count = current.shape[0] * current.shape[1]
    return l2_sum, l2_count, dot_sum, dot_count, cos_distance_sum, cos_distance_count


def autocast_context(config: TrainConfig, device: torch.device):
    if device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if config.bf16 else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=config.bf16 or config.fp16)


def compute_steps(config: TrainConfig, world_size: int) -> tuple[int, int]:
    tokens_per_step = config.batch_size * config.seq_len * config.grad_accum_steps * world_size
    max_steps = math.ceil(config.target_tokens / tokens_per_step)
    return tokens_per_step, max_steps


def get_lr_scheduler(config: TrainConfig, optimizer, max_steps: int, tokens_per_step: int):
    warmup_steps = config.warmup_steps or math.ceil(config.warmup_tokens / tokens_per_step)
    if config.lr_scheduler == "constant":
        return None
    lr_decay_steps = max_steps
    if config.lr_decay_tokens is not None:
        lr_decay_steps = math.ceil(config.lr_decay_tokens / tokens_per_step)
    min_ratio = config.min_learning_rate / config.learning_rate

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, step / warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, lr_decay_steps - warmup_steps))
        if config.lr_scheduler == "cosine":
            value = min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))
        elif config.lr_scheduler == "linear":
            value = min_ratio + (1.0 - min_ratio) * (1.0 - progress)
        else:
            raise ValueError(f"Unknown lr_scheduler {config.lr_scheduler!r}")
        return max(min_ratio, value)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(
        model,
        loader,
        config: TrainConfig,
        device: torch.device,
        max_eval_batches: int,
        continuous_concept_lambda: float,
        world_size: int
) -> dict[str, float]:
    set_continuous_concept_lambda(model, continuous_concept_lambda)
    model.eval()
    losses: list[float] = []
    lm_losses: list[float] = []
    continuous_concept_losses: list[float] = []
    layer_step_mse_sums: list[float] = []
    layer_cosine_sums: list[float] = []
    layer_step_counts: list[int] = []
    layer_cosine_counts: list[int] = []
    continuous_concept_enabled = normalized_continuous_concept_mode(config) != "none"
    continuous_concept_l2_sum = 0.0
    continuous_concept_l2_count = 0
    continuous_concept_dot_sum = 0.0
    continuous_concept_dot_count = 0
    continuous_concept_cos_sum = 0.0
    continuous_concept_cos_count = 0
    for i, batch in enumerate(loader):
        if i >= max_eval_batches:
            break
        batch = move_batch(batch, device)
        with autocast_context(config, device):
            outputs = model(**batch, output_hidden_states=True)

        loss = output_value(outputs, "loss")
        lm_loss = output_value(outputs, "lm_loss")
        continuous_concept_loss = output_value(outputs, "continuous_concept_loss")

        loss = reduce_mean(loss, world_size)
        lm_loss = reduce_mean(lm_loss, world_size)
        continuous_concept_loss = reduce_mean(continuous_concept_loss, world_size)

        losses.append(float(loss.detach().cpu()))
        lm_losses.append(float(lm_loss.detach().cpu()))
        continuous_concept_losses.append(float(continuous_concept_loss.detach().cpu()))

        hidden_states = output_value(outputs, "hidden_states")
        if hidden_states is None:
            raise RuntimeError("Eval hidden states were requested but the model did not return them")
        if not layer_step_mse_sums:
            num_layers = len(hidden_states)
            layer_step_mse_sums = [0.0] * num_layers
            layer_cosine_sums = [0.0] * num_layers
            layer_step_counts = [0] * num_layers
            layer_cosine_counts = [0] * num_layers
        for layer_idx, hidden in enumerate(hidden_states):
            step_sum, cosine_sum, step_count, cosine_count = hidden_layer_eval_sums(hidden.detach().float())
            layer_step_mse_sums[layer_idx] += float(step_sum.cpu())
            layer_cosine_sums[layer_idx] += float(cosine_sum.cpu())
            layer_step_counts[layer_idx] += step_count
            layer_cosine_counts[layer_idx] += cosine_count
        if continuous_concept_enabled:
            hidden = select_continuous_concept_hidden_state(hidden_states,
                                                            config.continuous_concept_layer).detach().float()
            l2_sum, l2_count, dot_sum, dot_count, cos_sum, cos_count = continuous_concept_eval_sums(hidden)
            continuous_concept_l2_sum += float(l2_sum.cpu())
            continuous_concept_l2_count += l2_count
            continuous_concept_dot_sum += float(dot_sum.cpu())
            continuous_concept_dot_count += dot_count
            continuous_concept_cos_sum += float(cos_sum.cpu())
            continuous_concept_cos_count += cos_count
    model.train()
    if not losses:
        metrics = {
            "val_loss": float("nan"),
            "val_lm_loss": float("nan"),
            "val_continuous_concept_loss": float("nan"),
            "val_continuous_concept_lambda": continuous_concept_lambda,
            "val_ppl": float("nan"),
        }
        if continuous_concept_enabled:
            metrics.update(
                {
                    "continuous_concept_l2": float("nan"),
                    "continuous_concept_dot": float("nan"),
                    "continuous_concept_cos": float("nan"),
                }
            )
        return metrics
    val_lm_loss = float(np.mean(lm_losses))
    metrics = {
        "val_loss": float(np.mean(losses)),
        "val_lm_loss": val_lm_loss,
        "val_continuous_concept_loss": float(np.mean(continuous_concept_losses)),
        "val_continuous_concept_lambda": continuous_concept_lambda,
        "val_ppl": float(math.exp(min(20.0, val_lm_loss))),
    }
    if continuous_concept_enabled:
        l2_sum = reduce_sum(continuous_concept_l2_sum, world_size, device).item()
        l2_count = reduce_sum(continuous_concept_l2_count, world_size, device).item()
        dot_sum = reduce_sum(continuous_concept_dot_sum, world_size, device).item()
        dot_count = reduce_sum(continuous_concept_dot_count, world_size, device).item()
        cos_sum = reduce_sum(continuous_concept_cos_sum, world_size, device).item()
        cos_count = reduce_sum(continuous_concept_cos_count, world_size, device).item()
        metrics["continuous_concept_l2"] = l2_sum / l2_count if l2_count > 0 else float("nan")
        metrics["continuous_concept_dot"] = dot_sum / dot_count if dot_count > 0 else float("nan")
        metrics["continuous_concept_cos"] = cos_sum / cos_count if cos_count > 0 else float("nan")
    for layer_idx in range(len(layer_step_mse_sums)):
        step_sum = reduce_sum(layer_step_mse_sums[layer_idx], world_size, device).item()
        step_count = reduce_sum(layer_step_counts[layer_idx], world_size, device).item()
        cosine_sum = reduce_sum(layer_cosine_sums[layer_idx], world_size, device).item()
        cosine_count = reduce_sum(layer_cosine_counts[layer_idx], world_size, device).item()
        if step_count > 0:
            metrics[f"layer_{layer_idx}/step_mse"] = step_sum / step_count
        if cosine_count > 0:
            metrics[f"layer_{layer_idx}/adjacent_cosine"] = cosine_sum / cosine_count
    return metrics


def save_checkpoint(
        path: Path,
        model,
        optimizer,
        scheduler,
        config: TrainConfig,
        step: int,
        trained_tokens: int,
        meta: dict[str, Any],
) -> None:
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "config": config.to_dict(),
            "step": step,
            "trained_tokens": trained_tokens,
            "meta": meta,
        },
        path / "checkpoint.pt",
    )


def maybe_resume(path: str | None, model, optimizer, scheduler, device: torch.device) -> tuple[int, int]:
    if not path:
        return 0, 0
    checkpoint = torch.load(path, map_location=device)
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint.get("step", 0)), int(checkpoint.get("trained_tokens", 0))


def main() -> None:
    config = parse_args()
    distributed, rank, world_size, local_rank = setup_distributed()
    seed_everything(config.seed + rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config.output_dir)
    data_dir = Path(config.data_dir)
    meta = load_meta(data_dir)

    vocab_size = int(meta["vocab_size"])

    tokens_per_step, max_steps = compute_steps(config, world_size)
    train_loader = make_loader(
        data_dir / "train.bin",
        config.seq_len,
        meta["dtype"],
        config.batch_size,
        config.num_workers,
        True,
        config.seed,
        distributed,
    )

    val_loader = None
    final_val_loader = None
    if (data_dir / "val.bin").exists():
        eval_batches = max(1, math.ceil(config.eval_tokens / (config.batch_size * config.seq_len * max(1, world_size))))
        val_loader = make_loader(
            data_dir / "val.bin",
            config.seq_len,
            meta["dtype"],
            config.batch_size,
            config.num_workers,
            False,
            config.seed,
            distributed,
        )
        if config.eval_tokens_final is not None:
            final_eval_tokens = config.eval_tokens_final
            final_eval_batches = max(
                1,
                math.ceil(final_eval_tokens / (config.batch_size * config.seq_len * max(1, world_size))),
            )
            final_val_loader = make_loader(
                data_dir / "val.bin",
                config.seq_len,
                meta["dtype"],
                config.batch_size,
                config.num_workers,
                False,
                config.seed,
                distributed,
            )
    else:
        eval_batches = 0
        final_eval_batches = 0

    model = build_model(config, vocab_size=vocab_size, eos_token_id=int(meta["eos_token_id"]), world_size=world_size)
    model.to(device)
    verify_bottleneck_attention_startup(model, config, rank)
    if config.torch_compile:
        if hasattr(model, "transformer"):
            model.transformer = torch.compile(model.transformer)
        else:
            model = torch.compile(model)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim < 2 or name.endswith(".bias") else decay).append(param)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": config.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
    )
    scheduler = get_lr_scheduler(config, optimizer, max_steps, tokens_per_step)
    start_step, trained_tokens = maybe_resume(config.resume_from, model, optimizer, scheduler, device)

    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16 and device.type == "cuda")
    if is_main(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config.json").open("w") as f:
            json.dump(config.to_dict(), f, indent=2)
            f.write("\n")
        print(f"training for up to {max_steps:,} optimizer steps ({config.target_tokens:,} text tokens)")
    wandb_run = setup_wandb(config, rank)

    set_continuous_concept_lambda(model, continuous_concept_lambda_at_tokens(config, trained_tokens))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    last_log = time.time()
    progress = None
    if is_main(rank):
        progress = tqdm(
            total=max_steps,
            initial=start_step,
            unit="step",
            dynamic_ncols=True,
            smoothing=0.05,
            desc="training",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

    try:
        while step < max_steps and trained_tokens < config.target_tokens:
            for micro_idx, batch in enumerate(train_loader):
                current_continuous_concept_lambda = continuous_concept_lambda_at_tokens(config, trained_tokens)
                set_continuous_concept_lambda(model, current_continuous_concept_lambda)
                batch = move_batch(batch, device)
                with autocast_context(config, device):
                    outputs = model(**batch)
                    loss = output_value(outputs, "loss")
                    loss = loss / config.grad_accum_steps

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (micro_idx + 1) % config.grad_accum_steps != 0:
                    continue

                if config.grad_clip > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

                step += 1
                trained_tokens += tokens_per_step

                current_loss = loss.detach()
                current_lm_loss = output_value(outputs, "lm_loss").detach()
                current_continuous_concept_loss = output_value(outputs, "continuous_concept_loss").detach()

                current_loss = reduce_mean(current_loss, world_size)
                current_lm_loss = reduce_mean(current_lm_loss, world_size)
                current_continuous_concept_loss = reduce_mean(current_continuous_concept_loss, world_size)

                current_loss = float(current_loss.cpu()) * config.grad_accum_steps
                current_lm_loss = float(current_lm_loss.cpu())
                current_continuous_concept_loss = float(current_continuous_concept_loss.cpu())

                current_lr = optimizer.param_groups[0]["lr"]
                if wandb_run is not None:
                    train_metrics = {
                        "train/loss": current_loss,
                        "train/lm_loss": current_lm_loss,
                        "train/continuous_concept_loss": current_continuous_concept_loss,
                        "train/continuous_concept_lambda": current_continuous_concept_lambda,
                        "train/lr": current_lr,
                        "train/tokens": trained_tokens,
                        "train/step": step,
                    }
                    if normalized_bottleneck_mode(config) is not None:
                        train_metrics.update(
                            {
                                "debug/num_concepts": float(output_value(outputs, "num_concepts").detach().cpu()),
                                "debug/seq_len": config.seq_len,
                                "debug/concept_k": config.bottleneck_concept_k,
                                "debug/bottleneck_mode": config.bottleneck_mode,
                                "debug/bottleneck_layers": config.bottleneck_layers,
                                "debug/attention_allowed_density": float(
                                    output_value(outputs, "attention_allowed_density").detach().cpu()
                                ),
                                "debug/continuous_concept_mode": config.continuous_concept_mode,
                                "debug/continuous_concept_lambda": config.continuous_concept_lambda,
                                "debug/continuous_concept_layer": config.continuous_concept_layer,
                            }
                        )
                    wandb_run.log(train_metrics, step=step)
                if progress is not None:
                    progress.update(1)
                    if step % config.log_interval == 0 or step == 1:
                        now = time.time()
                        progress.set_postfix(
                            {
                                "tokens": f"{trained_tokens:,}",
                                "loss": f"{current_loss:.4f}",
                                "lm": f"{current_lm_loss:.4f}",
                                "cc": f"{current_continuous_concept_loss:.4f}",
                                "ccl": f"{current_continuous_concept_lambda:.2e}",
                                "lr": f"{current_lr:.3e}",
                                "dt": f"{now - last_log:.1f}s",
                            },
                            refresh=True,
                        )
                        last_log = now

                should_eval = (
                        val_loader is not None
                        and config.eval_interval_tokens > 0
                        and trained_tokens % config.eval_interval_tokens < tokens_per_step
                )
                if should_eval:
                    eval_continuous_concept_lambda = continuous_concept_lambda_at_tokens(config, trained_tokens)
                    metrics = evaluate(
                        model,
                        val_loader,
                        config,
                        device,
                        eval_batches,
                        eval_continuous_concept_lambda,
                        world_size
                    )
                    if is_main(rank):
                        tqdm.write(
                            f"eval tokens {trained_tokens:,} val_loss {metrics['val_loss']:.4f} "
                            f"val_lm_loss {metrics['val_lm_loss']:.4f} val_ppl {metrics['val_ppl']:.2f} "
                            f"val_cc_loss {metrics['val_continuous_concept_loss']:.4f}"
                        )
                    if wandb_run is not None:
                        eval_metrics = {
                            "eval/loss": metrics["val_loss"],
                            "eval/lm_loss": metrics["val_lm_loss"],
                            "eval/continuous_concept_loss": metrics["val_continuous_concept_loss"],
                            "eval/continuous_concept_lambda": metrics["val_continuous_concept_lambda"],
                            "eval/ppl": metrics["val_ppl"],
                            "eval/tokens": trained_tokens,
                        }
                        for key, value in metrics.items():
                            if key.startswith("layer_"):
                                eval_metrics[f"eval/{key}"] = value
                        for key in ("continuous_concept_l2", "continuous_concept_dot", "continuous_concept_cos"):
                            if key in metrics:
                                eval_metrics[f"eval/{key}"] = metrics[key]
                        wandb_run.log(eval_metrics, step=step)

                should_save = config.save_interval_tokens > 0 and trained_tokens % config.save_interval_tokens < tokens_per_step
                if is_main(rank) and should_save:
                    save_checkpoint(output_dir / f"step_{step:08d}", model, optimizer, scheduler, config, step,
                                    trained_tokens, meta)

                if step >= max_steps or trained_tokens >= config.target_tokens:
                    break
    finally:
        if progress is not None:
            progress.close()

    if final_val_loader is not None:
        final_metrics = evaluate(
            model,
            final_val_loader,
            config,
            device,
            final_eval_batches,
            continuous_concept_lambda_at_tokens(config, trained_tokens),
            world_size,
        )
        if is_main(rank):
            print(
                f"final eval tokens {trained_tokens:,} val_lm_loss {final_metrics['val_lm_loss']:.4f} "
                f"({final_eval_tokens:,} validation tokens)"
            )
        if wandb_run is not None:
            wandb_run.log({"final_lm_loss": final_metrics["val_lm_loss"]}, step=step)

    if is_main(rank):
        save_checkpoint(output_dir / "final", model, optimizer, scheduler, config, step, trained_tokens, meta)
        print(f"finished at step {step:,}, trained_tokens={trained_tokens:,}")
    if wandb_run is not None:
        wandb_run.finish()
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()

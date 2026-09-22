"""Evaluation-only representation diagnostics for matched BBC and CC checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import fields
from itertools import islice
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from config import TrainConfig
from model import build_model, gpt2_blocks, parse_bottleneck_layers, select_continuous_concept_hidden_state
from train import make_loader, move_batch


SPECTRAL_CUTOFFS = (0.05, 0.10, 0.20, 0.30, 0.50)


def unwrap_compiled_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove torch.compile wrapper names from checkpoint keys."""
    return {key.replace("._orig_mod.", ".").removeprefix("_orig_mod."): value for key, value in state_dict.items()}


def load_checkpoint_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    config_keys = {field.name for field in fields(TrainConfig)}
    config = TrainConfig(**{key: value for key, value in checkpoint["config"].items() if key in config_keys})
    meta = checkpoint.get("meta")
    if meta is None:
        raise ValueError(f"Checkpoint {path} does not contain dataset metadata")
    model = build_model(config, vocab_size=int(meta["vocab_size"]), eos_token_id=int(meta["eos_token_id"]))
    model.load_state_dict(unwrap_compiled_state_dict(checkpoint["model"]))
    model.to(device).eval()
    return model, config, meta


def validate_matched_configs(first: TrainConfig, second: TrainConfig) -> None:
    if first.seq_len != second.seq_len:
        raise ValueError(f"Checkpoint sequence lengths differ: {first.seq_len} vs {second.seq_len}")


def validate_matched_models(first, second) -> None:
    first_config = first.transformer.config
    second_config = second.transformer.config
    if (first_config.n_layer, first_config.n_head, first_config.n_embd) != (
        second_config.n_layer,
        second_config.n_head,
        second_config.n_embd,
    ):
        raise ValueError("Checkpoint model dimensions differ; use matched checkpoints")


def per_sequence_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return one mean next-token NLL per sequence, ignoring masked labels."""
    batch_size, sequence_length, vocab_size = logits.shape
    token_nll = F.cross_entropy(
        logits.reshape(batch_size * sequence_length, vocab_size),
        labels.reshape(batch_size * sequence_length),
        ignore_index=-100,
        reduction="none",
    ).view(batch_size, sequence_length)
    valid = labels.ne(-100)
    return (token_nll * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def replace_completed_segment_states(
    hidden: torch.Tensor,
    concept_k: int,
    condition: str,
    non_boundary_indices: torch.Tensor | None = None,
    mode: str = "zero",
    mean_window: int | None = None,
    warmup_segments: int = 0,
) -> torch.Tensor:
    """Replace one selected state per completed segment without future information."""
    if condition == "full":
        return hidden
    if condition not in {"boundary_removed", "random_non_boundary_removed"}:
        raise ValueError(f"Unknown BBC intervention condition: {condition}")
    if concept_k <= 0:
        raise ValueError("concept_k must be positive")
    if mode not in {"zero", "mean"}:
        raise ValueError(f"Unknown BBC replacement mode: {mode}")
    if mode == "mean" and (mean_window is None or mean_window <= 0):
        raise ValueError("mean replacement requires a positive mean_window")
    if warmup_segments < 0:
        raise ValueError("warmup_segments must be non-negative")

    completed_length = hidden.shape[1] // concept_k * concept_k
    if completed_length == 0:
        return hidden
    num_segments = completed_length // concept_k
    if warmup_segments >= num_segments:
        return hidden
    active_segments = torch.arange(warmup_segments, num_segments, device=hidden.device)
    if condition == "boundary_removed":
        selected_indices = torch.full(
            (hidden.shape[0], num_segments), concept_k - 1, device=hidden.device, dtype=torch.long
        )
    else:
        if concept_k < 2:
            raise ValueError("random_non_boundary_removed requires concept_k of at least 2")
        if non_boundary_indices is None:
            raise ValueError("random_non_boundary_removed requires non_boundary_indices")
        expected_shape = (hidden.shape[0], num_segments)
        if non_boundary_indices.shape != expected_shape:
            raise ValueError(
                f"non_boundary_indices must have shape {expected_shape}, got {non_boundary_indices.shape}"
            )
        if not torch.all((0 <= non_boundary_indices) & (non_boundary_indices < concept_k - 1)):
            raise ValueError("non_boundary_indices must select non-boundary positions")
        selected_indices = non_boundary_indices.to(device=hidden.device, dtype=torch.long)

    selected_indices = selected_indices[:, warmup_segments:]
    segment_starts = active_segments[None, :] * concept_k
    positions = segment_starts + selected_indices
    if mode == "zero":
        replacement = torch.zeros(
            (*selected_indices.shape, hidden.shape[-1]), dtype=hidden.dtype, device=hidden.device
        )
    else:
        start = (positions - mean_window).clamp_min(0)
        prefix = torch.cat(
            [torch.zeros_like(hidden[:, :1, :], dtype=torch.float32), hidden.float().cumsum(dim=1)], dim=1
        )
        gather_index = positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        start_index = start.unsqueeze(-1).expand_as(gather_index)
        previous_sum = prefix.gather(1, gather_index) - prefix.gather(1, start_index)
        previous_count = (positions - start).clamp_min(1).unsqueeze(-1)
        replacement = (previous_sum / previous_count).to(hidden.dtype)

    modified = hidden.clone()
    completed = modified[:, :completed_length, :].view(hidden.shape[0], num_segments, concept_k, hidden.shape[-1])
    batch_indices = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
    segment_indices = active_segments[None, :]
    completed[batch_indices, segment_indices, selected_indices, :] = replacement
    return modified


def with_block_output_intervention(model, block_index: int, intervention: Callable[[torch.Tensor], torch.Tensor]):
    """Attach a temporary hook that changes a GPT-2 block output before its successor."""
    blocks = gpt2_blocks(model.transformer)
    if not 0 <= block_index < len(blocks):
        raise ValueError(f"BBC block index {block_index} is out of range for {len(blocks)} layers")

    def hook(_module, _inputs, output):
        if not isinstance(output, tuple):
            raise RuntimeError("Expected GPT-2 block output to be a tuple")
        return (intervention(output[0]), *output[1:])

    return blocks[block_index].register_forward_hook(hook)


def resolve_bbc_block_for_model(layer: str | None, config: TrainConfig, model) -> int:
    n_layers = len(gpt2_blocks(model.transformer))
    bottleneck_layers = parse_bottleneck_layers(config.bottleneck_layers, n_layers)
    if layer is not None:
        layers = parse_bottleneck_layers(layer, n_layers)
        if len(layers) != 1:
            raise ValueError("--layer must identify exactly one BBC layer")
        if layers[0] not in bottleneck_layers:
            raise ValueError("--layer must be a layer where BBC was applied during training")
        return layers[0]
    if len(bottleneck_layers) != 1:
        raise ValueError("BBC checkpoint uses multiple bottleneck layers; specify --layer")
    return bottleneck_layers[0]


def parse_bbc_intervention_mode(mode: str, concept_k: int) -> tuple[str, int | None, int | None]:
    """Return replacement mode, mean window, and fixed interior distance from the boundary."""
    name, separator, value = mode.partition(":")
    if name not in {"zero", "mean"}:
        raise ValueError("--mode must be zero, zero:k, mean, or mean:k")
    if not separator:
        return ("zero", None, None) if name == "zero" else ("mean", concept_k - 1, None)
    try:
        amount = int(value)
    except ValueError as exc:
        raise ValueError("The :k portion of --mode must be a positive integer") from exc
    if amount <= 0:
        raise ValueError("The :k portion of --mode must be a positive integer")
    if name == "mean":
        return name, amount, None
    if amount >= concept_k:
        raise ValueError(f"zero:k requires k < concept_k ({concept_k})")
    return name, None, amount


def continuous_hidden_state_index(layer: str, hidden_state_count: int) -> int:
    if layer.lower() in {"mid", "middle"}:
        return hidden_state_count // 2
    if layer.lower() in {"last", "final"}:
        return hidden_state_count - 1
    try:
        index = int(layer)
    except ValueError as exc:
        raise ValueError("--layer must be mid, last, or a hidden-state index") from exc
    if not -hidden_state_count <= index < hidden_state_count:
        raise ValueError(f"Hidden-state index {index} is out of range")
    return index % hidden_state_count


def spectral_basis(sequence_length: int, device: torch.device) -> torch.Tensor:
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    laplacian = torch.zeros((sequence_length, sequence_length), dtype=torch.float64)
    diagonal = torch.full((sequence_length,), 2.0, dtype=torch.float64)
    if sequence_length > 1:
        diagonal[0] = diagonal[-1] = 1.0
        off_diagonal = torch.full((sequence_length - 1,), -1.0, dtype=torch.float64)
        laplacian.diagonal().copy_(diagonal)
        laplacian.diagonal(1).copy_(off_diagonal)
        laplacian.diagonal(-1).copy_(off_diagonal)
    else:
        laplacian[0, 0] = 0.0
    _, eigenvectors = torch.linalg.eigh(laplacian)
    return eigenvectors.to(device=device, dtype=torch.float32)


def spectral_metrics(hidden: torch.Tensor, basis: torch.Tensor, cutoffs: tuple[float, ...] = SPECTRAL_CUTOFFS) -> dict[str, torch.Tensor]:
    hidden = hidden.float()
    total_energy = hidden.square().sum(dim=(1, 2))
    metrics: dict[str, torch.Tensor] = {"hidden_energy": total_energy}
    if hidden.shape[1] > 1:
        raw_energy = (hidden[:, 1:, :] - hidden[:, :-1, :]).square().sum(dim=(1, 2))
    else:
        raw_energy = torch.zeros_like(total_energy)
    metrics["dirichlet_energy"] = raw_energy
    metrics["normalized_dirichlet_energy"] = raw_energy / total_energy.clamp_min(1e-12)
    for cutoff in cutoffs:
        m = max(1, min(hidden.shape[1], round(cutoff * hidden.shape[1])))
        coefficients = torch.einsum("nm,bnd->bmd", basis[:, :m], hidden)
        residual = (total_energy - coefficients.square().sum(dim=(1, 2))).clamp_min(0.0)
        metrics[f"residual_{m}"] = residual / total_energy.clamp_min(1e-12)
    return metrics


def mean_and_standard_error(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "standard_error": float("nan"), "count": 0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
    return {"mean": mean, "standard_error": math.sqrt(variance / len(values)), "count": len(values)}


def beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the continued fraction used by the regularized incomplete beta."""
    max_iterations = 200
    epsilon = 3e-14
    minimum = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = max(d, minimum)
    d = 1.0 / d
    result = d
    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        coefficient = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = max(1.0 + coefficient * d, minimum)
        c = max(1.0 + coefficient / c, minimum)
        d = 1.0 / d
        result *= d * c
        coefficient = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = max(1.0 + coefficient * d, minimum)
        c = max(1.0 + coefficient / c, minimum)
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            return result
    raise RuntimeError("Incomplete-beta continued fraction did not converge")


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Return I_x(a, b) without requiring SciPy."""
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must be between 0 and 1")
    if x == 0.0 or x == 1.0:
        return x
    front = math.exp(a * math.log(x) + b * math.log1p(-x) + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * beta_continued_fraction(a, b, x) / a
    return 1.0 - front * beta_continued_fraction(b, a, 1.0 - x) / b


def paired_t_test(first: list[float], second: list[float]) -> dict[str, float]:
    """Two-sided paired Student t-test for equally sized paired samples."""
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("Paired t-test requires at least two equally sized samples")
    differences = [right - left for left, right in zip(first, second)]
    mean = sum(differences) / len(differences)
    variance = sum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
    standard_error = math.sqrt(variance / len(differences))
    if standard_error == 0.0:
        t_statistic = math.inf if mean != 0.0 else 0.0
        p_value = 0.0 if mean != 0.0 else 1.0
    else:
        t_statistic = mean / standard_error
        degrees_of_freedom = len(differences) - 1
        x = degrees_of_freedom / (degrees_of_freedom + t_statistic**2)
        p_value = regularized_incomplete_beta(degrees_of_freedom / 2.0, 0.5, x)
    return {"mean": mean, "standard_error": standard_error, "t_statistic": t_statistic, "p_value": p_value,
            "degrees_of_freedom": len(differences) - 1}


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def run_bbc_diagnostic(args) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vanilla, vanilla_config, vanilla_meta = load_checkpoint_model(Path(args.vanilla_checkpoint), device)
    bbc, bbc_config, bbc_meta = load_checkpoint_model(Path(args.bbc_checkpoint), device)
    validate_matched_configs(vanilla_config, bbc_config)
    if vanilla_config.concept_k != bbc_config.concept_k:
        raise ValueError(
            f"Checkpoint concept lengths differ: {vanilla_config.concept_k} vs {bbc_config.concept_k}"
        )
    if vanilla_config.concept_k < 2:
        raise ValueError("BBC random non-boundary control requires concept_k of at least 2")
    replacement_mode, mean_window, interior_distance = parse_bbc_intervention_mode(
        args.mode, vanilla_config.concept_k
    )
    if args.warmup_segments < 0:
        raise ValueError("warmup_segments must be non-negative")
    if args.warmup_segments * vanilla_config.concept_k >= vanilla_config.seq_len:
        raise ValueError("warmup_segments leaves no tokens to score")
    validate_matched_models(vanilla, bbc)
    if int(vanilla_meta["vocab_size"]) != int(bbc_meta["vocab_size"]):
        raise ValueError("Checkpoint vocabularies differ")
    block_index = resolve_bbc_block_for_model(args.layer, bbc_config, bbc)
    if block_index >= len(gpt2_blocks(vanilla.transformer)):
        raise ValueError("Vanilla checkpoint does not have the requested BBC layer")
    loader = make_loader(
        Path(args.data_dir) / "val.bin",
        vanilla_config.seq_len,
        vanilla_meta["dtype"],
        args.batch_size,
        args.num_workers,
        False,
        args.seed,
        False,
    )
    rows: list[dict[str, float | int | str]] = []
    contrasts = {"vanilla": [], "bbc": []}
    conditions = ("full", "boundary_removed", "random_non_boundary_removed")
    random_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    for batch_index, batch in enumerate(
        tqdm(islice(loader, args.num_batches), total=args.num_batches, desc="BBC diagnostic", unit="batch")
    ):
        batch = move_batch(batch, device)
        scored_labels = batch["labels"].clone()
        scored_labels[:, : args.warmup_segments * vanilla_config.concept_k] = -100
        completed_segments = batch["input_ids"].shape[1] // vanilla_config.concept_k
        if interior_distance is None:
            non_boundary_indices = torch.randint(
                vanilla_config.concept_k - 1,
                (batch["input_ids"].shape[0], completed_segments),
                generator=random_generator,
                dtype=torch.long,
            )
        else:
            non_boundary_indices = torch.full(
                (batch["input_ids"].shape[0], completed_segments),
                vanilla_config.concept_k - 1 - interior_distance,
                dtype=torch.long,
            )
        for name, model, config in (("vanilla", vanilla, vanilla_config), ("bbc", bbc, bbc_config)):
            values: dict[str, torch.Tensor] = {}
            for condition in conditions:
                if condition == "full":
                    outputs = model(**batch)
                else:
                    handle = with_block_output_intervention(
                        model, block_index,
                        lambda hidden, c=condition: replace_completed_segment_states(
                            hidden,
                            config.concept_k,
                            c,
                            non_boundary_indices,
                            replacement_mode,
                            mean_window,
                            args.warmup_segments,
                        ),
                    )
                    try:
                        outputs = model(**batch)
                    finally:
                        handle.remove()
                values[condition] = per_sequence_nll(outputs.logits, scored_labels).float().cpu()
            for example_index in range(batch["input_ids"].shape[0]):
                full = float(values["full"][example_index])
                removed = float(values["boundary_removed"][example_index])
                interior_removed = float(values["random_non_boundary_removed"][example_index])
                boundary_minus_interior = removed - interior_removed
                contrasts[name].append(boundary_minus_interior)
                rows.append({"model": name, "batch": batch_index, "example": example_index, "full_nll": full,
                             "boundary_removed_nll": removed, "delta_boundary_remove": removed - full,
                             "interior_removed_nll": interior_removed,
                             "delta_interior_remove": interior_removed - full,
                             "delta_boundary_minus_interior": boundary_minus_interior})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "bbc_boundary_sufficiency.csv", rows)
    summary = {name: {metric: mean_and_standard_error([float(row[metric]) for row in rows if row["model"] == name])
                      for metric in (
                          "full_nll", "boundary_removed_nll", "delta_boundary_remove",
                          "interior_removed_nll", "delta_interior_remove", "delta_boundary_minus_interior",
                      )}
               for name in ("vanilla", "bbc")}
    (args.output_dir / "bbc_boundary_sufficiency_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("model   delta_b   delta_i   delta_b-delta_i")
    for name in ("vanilla", "bbc"):
        def format_mean_std(metrics: dict[str, float]) -> str:
            standard_deviation = metrics["standard_error"] * math.sqrt(metrics["count"])
            return f"{metrics['mean']:.4f}±{standard_deviation:.4f}"

        print(
            f"{name:<7} {format_mean_std(summary[name]['delta_boundary_remove']):>21} "
            f"{format_mean_std(summary[name]['delta_interior_remove']):>21} "
            f"{format_mean_std(summary[name]['delta_boundary_minus_interior']):>31}"
        )
    test = paired_t_test(contrasts["vanilla"], contrasts["bbc"])
    print(
        f"paired BBC-vanilla contrast: t({int(test['degrees_of_freedom'])})={test['t_statistic']:.3f}, "
        f"p={test['p_value']:.3g}"
    )


@torch.inference_mode()
def run_cc_diagnostic(args) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vanilla, vanilla_config, vanilla_meta = load_checkpoint_model(Path(args.vanilla_checkpoint), device)
    cc, cc_config, cc_meta = load_checkpoint_model(Path(args.cc_checkpoint), device)
    validate_matched_configs(vanilla_config, cc_config)
    validate_matched_models(vanilla, cc)
    if vanilla_config.seq_len != cc_config.seq_len or int(vanilla_meta["vocab_size"]) != int(cc_meta["vocab_size"]):
        raise ValueError("Checkpoint sequence lengths or vocabularies differ")
    layer = args.layer or cc_config.continuous_concept_layer
    hidden_state_count = len(gpt2_blocks(cc.transformer)) + 1
    if continuous_hidden_state_index(layer, hidden_state_count) != continuous_hidden_state_index(
        cc_config.continuous_concept_layer, hidden_state_count
    ):
        raise ValueError("--layer must be the layer where CC was applied during training")
    loader = make_loader(
        Path(args.data_dir) / "val.bin",
        vanilla_config.seq_len,
        vanilla_meta["dtype"],
        args.batch_size,
        args.num_workers,
        False,
        args.seed,
        False,
    )
    basis = spectral_basis(vanilla_config.seq_len, device)
    rows: list[dict[str, float | int | str]] = []
    for batch_index, batch in enumerate(
        tqdm(islice(loader, args.num_batches), total=args.num_batches, desc="CC diagnostic", unit="batch")
    ):
        batch = move_batch(batch, device)
        for name, model in (("vanilla", vanilla), ("cc", cc)):
            outputs = model(**batch, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden states")
            hidden = select_continuous_concept_hidden_state(hidden_states, layer)
            metrics = spectral_metrics(hidden, basis)
            for example_index in range(hidden.shape[0]):
                row: dict[str, float | int | str] = {"model": name, "batch": batch_index, "example": example_index,
                                                      "dirichlet_energy": float(metrics["dirichlet_energy"][example_index].cpu()),
                                                      "normalized_dirichlet_energy": float(metrics["normalized_dirichlet_energy"][example_index].cpu()),
                                                      "hidden_energy": float(metrics["hidden_energy"][example_index].cpu())}
                for cutoff in SPECTRAL_CUTOFFS:
                    m = max(1, min(hidden.shape[1], round(cutoff * hidden.shape[1])))
                    row[f"residual_m{m}"] = float(metrics[f"residual_{m}"][example_index].cpu())
                rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "cc_spectral_residual.csv", rows)
    summary = {name: {metric: mean_and_standard_error([float(row[metric]) for row in rows if row["model"] == name])
                      for metric in list(rows[0]) if metric.startswith("residual_") or metric in {"dirichlet_energy", "normalized_dirichlet_energy", "hidden_energy"}}
               for name in ("vanilla", "cc")}
    (args.output_dir / "cc_spectral_residual_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    from chart import plot_spectral_residual

    plot_spectral_residual(
        args.output_dir / "cc_spectral_residual.pdf",
        spectral_residual_series(summary, vanilla_config.seq_len),
    )
    print(json.dumps(summary, indent=2))


def spectral_residual_series(summary: dict, sequence_length: int) -> dict[str, dict[float, float]]:
    """Convert a diagnostic summary into the chart module's series format."""
    return {
        name.title(): {
            int(metric.removeprefix("residual_m")) / sequence_length: values["mean"]
            for metric, values in metrics.items()
            if metric.startswith("residual_m")
        }
        for name, metrics in summary.items()
    }


def run_correctness_tests() -> None:
    hidden = torch.arange(10.0).view(1, 10, 1)
    boundary_zero = replace_completed_segment_states(hidden, 4, "boundary_removed")
    assert torch.equal(boundary_zero[0, :, 0], torch.tensor([0.0, 1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 0.0, 8.0, 9.0]))
    boundary_zero_warmup = replace_completed_segment_states(hidden, 4, "boundary_removed", warmup_segments=1)
    assert torch.equal(boundary_zero_warmup[0, :, 0], torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 8.0, 9.0]))
    random_zero = replace_completed_segment_states(
        hidden, 4, "random_non_boundary_removed", torch.tensor([[0, 2]])
    )
    assert torch.equal(random_zero[0, :, 0], torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]))
    boundary_mean = replace_completed_segment_states(hidden, 4, "boundary_removed", mode="mean", mean_window=4)
    assert torch.equal(boundary_mean[0, :, 0], torch.tensor([0.0, 1.0, 2.0, 1.0, 4.0, 5.0, 6.0, 4.5, 8.0, 9.0]))
    random_mean = replace_completed_segment_states(
        hidden, 4, "random_non_boundary_removed", torch.tensor([[0, 2]]), mode="mean", mean_window=4
    )
    assert torch.equal(random_mean[0, :, 0], torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 3.5, 7.0, 8.0, 9.0]))
    boundary_last = replace_completed_segment_states(hidden, 4, "boundary_removed", mode="mean", mean_window=1)
    assert torch.equal(boundary_last[0, :, 0], torch.tensor([0.0, 1.0, 2.0, 2.0, 4.0, 5.0, 6.0, 6.0, 8.0, 9.0]))
    random_last = replace_completed_segment_states(
        hidden, 4, "random_non_boundary_removed", torch.tensor([[0, 2]]), mode="mean", mean_window=1
    )
    assert torch.equal(random_last[0, :, 0], torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 7.0, 8.0, 9.0]))
    assert parse_bbc_intervention_mode("zero", 4) == ("zero", None, None)
    assert parse_bbc_intervention_mode("zero:1", 4) == ("zero", None, 1)
    assert parse_bbc_intervention_mode("mean", 4) == ("mean", 3, None)
    assert parse_bbc_intervention_mode("mean:2", 4) == ("mean", 2, None)
    assert math.isclose(regularized_incomplete_beta(1.0, 1.0, 0.5), 0.5)
    test = paired_t_test([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
    assert test["mean"] > 0.0 and 0.0 < test["p_value"] < 1.0
    basis = spectral_basis(8, torch.device("cpu"))
    coefficients = torch.randn(1, 3, 2)
    low_frequency_hidden = torch.einsum("nm,bmd->bnd", basis[:, :3], coefficients)
    metrics = spectral_metrics(low_frequency_hidden, basis, cutoffs=(3 / 8, 4 / 8, 6 / 8))
    assert metrics["residual_3"].item() < 1e-5
    residuals = [metrics[f"residual_{m}"].item() for m in (3, 4, 6)]
    assert all(left + 1e-6 >= right for left, right in zip(residuals, residuals[1:]))
    print("representation diagnostic correctness tests passed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="diagnostic", required=False)
    for name in ("bbc", "cc"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--vanilla_checkpoint", required=True)
        subparser.add_argument(f"--{name}_checkpoint", required=True)
        subparser.add_argument("--data_dir", required=True)
        subparser.add_argument("--output_dir", type=Path, required=True)
        subparser.add_argument("--layer", default=None)
        subparser.add_argument("--num_batches", type=int, default=128)
        subparser.add_argument("--batch_size", type=int, default=8)
        subparser.add_argument("--num_workers", type=int, default=0)
        subparser.add_argument("--seed", type=int, default=82)
        subparser.add_argument("--device", default=None)
        if name == "bbc":
            subparser.add_argument(
                "--mode",
                default="zero",
                help="zero, zero:k, mean, or mean:k for matched single-token ablations.",
            )
            subparser.add_argument(
                "--warmup_segments",
                type=int,
                default=1,
                help="Initial segments excluded from intervention and NLL scoring.",
            )
    parser.add_argument("--run_tests", action="store_true")
    parser.add_argument("--plot_only", action="store_true", help="Regenerate a CC PDF from an existing summary JSON.")
    parser.add_argument("--summary_json", type=Path, help="Path to cc_spectral_residual_summary.json for --plot_only.")
    parser.add_argument("--sequence_length", type=int, help="Sequence length used to create the summary, for --plot_only.")
    parser.add_argument("--output_pdf", type=Path, help="Destination PDF path for --plot_only.")
    args = parser.parse_args()
    if args.plot_only:
        if args.summary_json is None or args.sequence_length is None or args.output_pdf is None:
            parser.error("--plot_only requires --summary_json, --sequence_length, and --output_pdf")
    elif not args.run_tests and args.diagnostic is None:
        parser.error("choose bbc or cc, or pass --run_tests")
    return args


def main() -> None:
    args = parse_args()
    if args.plot_only:
        with args.summary_json.open() as handle:
            summary = json.load(handle)
        args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
        from chart import plot_spectral_residual

        plot_spectral_residual(args.output_pdf, spectral_residual_series(summary, args.sequence_length))
    elif args.run_tests:
        run_correctness_tests()
    elif args.diagnostic == "bbc":
        run_bbc_diagnostic(args)
    else:
        run_cc_diagnostic(args)


if __name__ == "__main__":
    main()

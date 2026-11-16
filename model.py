from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel, GPTNeoXConfig, GPTNeoXForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXRotaryEmbedding

from config import TrainConfig, model_dims


@dataclass
class SCTOutput:
    loss: torch.Tensor
    lm_loss: torch.Tensor
    continuous_concept_loss: torch.Tensor
    logits: torch.Tensor
    num_concepts: torch.Tensor
    attention_allowed_density: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None


def lm_loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="mean",
    )


def normalize_bottleneck_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    lowered = mode.lower()
    if lowered in {"", "none", "null", "off", "false"}:
        return None
    if lowered in {"boundary", "sliding_window", "random_boundary", "random_sparse"}:
        return lowered
    raise ValueError("bottleneck_mode must be None, boundary, sliding_window, random_boundary, or random_sparse")


def model_blocks(transformer: GPT2LMHeadModel | GPTNeoXForCausalLM) -> nn.ModuleList:
    if isinstance(transformer, GPTNeoXForCausalLM):
        return transformer.gpt_neox.layers

    blocks = getattr(getattr(transformer, "transformer", None), "h", None)
    if blocks is None:
        raise RuntimeError("Could not find GPT-2 transformer blocks")

    return blocks


def parse_layer_spec(spec: str, n_layer: int, flag_name: str, label: str) -> list[int]:
    normalized = spec.strip().lower()
    if normalized == "all":
        return list(range(n_layer))
    if normalized == "last":
        return [n_layer - 1]
    if normalized == "mid":
        return [n_layer // 2]
    if not normalized:
        raise ValueError(f"--{flag_name} must be all, last, mid, an index, or comma-delimited indices")

    layers: list[int] = []
    for part in normalized.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"Invalid --{flag_name} value: {spec!r}")
        try:
            layer = int(part)
        except ValueError as exc:
            raise ValueError(f"Invalid {label} layer index {part!r}") from exc
        if layer < 0 or layer >= n_layer:
            raise ValueError(f"{label} layer index {layer} is out of range for {n_layer} layers")
        if layer not in layers:
            layers.append(layer)
    return layers


def parse_bottleneck_layers(spec: str, n_layer: int) -> list[int]:
    return parse_layer_spec(spec, n_layer, "bottleneck_layers", "bottleneck")


def patch_gpt2_attention_bias_layers(transformer: GPT2LMHeadModel, allow: torch.Tensor, layers: list[int],
                                     label: str) -> None:
    bias = allow[None, None, :, :]
    blocks = model_blocks(transformer)
    patched = 0
    for layer in layers:
        attn = getattr(blocks[layer], "attn", None)
        if attn is None or not hasattr(attn, "bias") or not isinstance(attn.bias, torch.Tensor):
            raise RuntimeError(f"Could not find GPT-2 attention bias for layer {layer}")
        attn.register_buffer("bias", bias.clone(), persistent=False)
        patched += 1
    if patched == 0:
        raise RuntimeError(f"Could not patch any GPT-2 attention bias buffers for {label}")


def build_boundary_concept_mask(
        batch_size: int,
        seq_len: int,
        concept_k: int,
        device: torch.device,
) -> torch.Tensor:
    if concept_k <= 0:
        raise ValueError("concept_k must be positive")
    pos = torch.arange(seq_len, device=device)
    is_concept = (pos % concept_k) == (concept_k - 1)
    return is_concept[None, :].expand(batch_size, seq_len)


def _random_choice_mask(seq_len: int, counts: torch.Tensor, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    allow = torch.zeros((seq_len, seq_len), dtype=torch.bool)
    for q in range(seq_len):
        count = min(int(counts[q].item()), q + 1)
        if count <= 0:
            continue
        selected = torch.randperm(q + 1, generator=generator)[:count]
        allow[q, selected] = True
    return allow.to(device=device)


def build_bottleneck_allow_mask(
        seq_len: int,
        concept_k: int,
        device: torch.device,
        mode: str | None,
        seed: int = 0,
) -> torch.Tensor:
    mode = normalize_bottleneck_mode(mode)
    if mode is None:
        return torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).tril()
    if concept_k <= 0:
        raise ValueError("concept_k must be positive")

    pos = torch.arange(seq_len, device=device)
    q = pos[:, None]
    k = pos[None, :]
    q_seg = q // concept_k
    k_seg = k // concept_k
    k_off = k % concept_k

    same_segment_causal = (k_seg == q_seg) & (k <= q)
    if mode == "boundary":
        k_is_concept = k_off == (concept_k - 1)
        previous_concepts = k_is_concept & (k_seg < q_seg)
        return same_segment_causal | previous_concepts

    counts = (pos // concept_k) + (pos % concept_k) + 1
    if mode == "sliding_window":
        window_start = (pos - counts + 1).clamp_min(0)
        return (k <= q) & (k >= window_start[:, None])

    if mode == "random_sparse":
        return _random_choice_mask(seq_len, counts.cpu(), seed, device)

    if mode == "random_boundary":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        num_segments = (seq_len + concept_k - 1) // concept_k
        offsets = torch.randint(0, concept_k, (num_segments,), generator=generator, device="cpu")
        seg_ids = torch.arange(num_segments, device="cpu")
        random_boundary_pos = seg_ids * concept_k + offsets
        random_boundary_pos = random_boundary_pos.clamp_max(seq_len - 1).to(device=device)
        k_is_random_boundary = torch.zeros(seq_len, dtype=torch.bool, device=device)
        k_is_random_boundary[random_boundary_pos] = True
        previous_concepts = k_is_random_boundary[None, :] & (k_seg < q_seg)
        return same_segment_causal | previous_concepts

    raise RuntimeError(f"Unhandled bottleneck_mode: {mode}")


def add_rope_changing_hook(transformer: GPTNeoXForCausalLM, config: TrainConfig) -> None:
    assert isinstance(transformer, GPTNeoXForCausalLM), 'Currently only GPTNeoX is supported'

    layers = transformer.gpt_neox.layers

    if not config.change_rope_layers:
        return

    rope_layers = config.change_rope_layers
    rope_layers = parse_layer_spec(rope_layers, len(layers), 'change_rope_layers', 'rope_layers')

    changed_model_config = copy.copy(transformer.config)
    changed_model_config.partial_rotary_factor = config.change_rope_rotary_pct
    changed_model_config.rotary_pct = config.change_rope_rotary_pct

    rope_emb = GPTNeoXRotaryEmbedding(changed_model_config)

    def change_rope_hook(_, args, kwargs):
        position_ids = kwargs['position_ids']
        hidden_state = args[0]
        kwargs['position_embeddings'] = rope_emb(hidden_state, position_ids)
        return args, kwargs

    for layer in rope_layers:
        layers[layer].register_forward_pre_hook(change_rope_hook, with_kwargs=True)



def add_bottleneck_hook(transformer: GPTNeoXForCausalLM, allow: torch.Tensor, layers: list[int]) -> None:
    min_dtype = torch.finfo(transformer.dtype).min

    bottleneck_mask = allow[None, None, :, :]
    bottleneck_mask = torch.where(bottleneck_mask, 0, min_dtype)

    def bottleneck_attention_hook(_, args, kwargs):
        kwargs["attention_mask"] = bottleneck_mask.to(transformer.device)
        return args, kwargs

    for layer in layers:
        transformer.gpt_neox.layers[layer].register_forward_pre_hook(bottleneck_attention_hook, with_kwargs=True)


def patch_bottleneck_attention(
        transformer: GPT2LMHeadModel | GPTNeoXForCausalLM,
        seq_len: int,
        concept_k: int,
        mode: str | None,
        bottleneck_layers: str = "all",
        seed: int = 0,
) -> float:
    device = next(transformer.parameters()).device
    allow = build_bottleneck_allow_mask(seq_len, concept_k, device=device, mode=mode, seed=seed)
    density = float(allow.float().mean().item())
    layers = parse_bottleneck_layers(bottleneck_layers, len(model_blocks(transformer)))

    if isinstance(transformer, GPT2LMHeadModel):
        patch_gpt2_attention_bias_layers(transformer, allow, layers, f"{mode} bottleneck attention")
    else:
        add_bottleneck_hook(transformer, allow, layers)

    return density


class VanillaGPT2LM(nn.Module):
    def __init__(
            self,
            transformer: GPT2LMHeadModel | GPTNeoXForCausalLM,
            concept_k: int = 8,
            continuous_concept_mode: str = "none",
            continuous_concept_lambda: float = 0.01,
            continuous_concept_layer: str = "mid",
            continuous_concept_dropout: float | None = None,
            continuous_concept_adaptive_dropout: float | None = None,
            continuous_concept_adaptive_dropout_history: int = 100,
            world_size: int = 1
    ):
        super().__init__()
        self.transformer = transformer
        self.concept_k = concept_k
        self.continuous_concept_mode = normalize_continuous_concept_mode(continuous_concept_mode)
        self.continuous_concept_lambda = continuous_concept_lambda
        self.continuous_concept_layer = continuous_concept_layer
        self.continuous_concept_dropout = validate_continuous_concept_dropout(
            self.continuous_concept_mode, continuous_concept_dropout
        )
        self.continuous_concept_adaptive_dropout = validate_continuous_concept_adaptive_dropout(
            self.continuous_concept_mode,
            continuous_concept_adaptive_dropout,
            continuous_concept_adaptive_dropout_history,
            self.continuous_concept_dropout,
        )
        if continuous_concept_adaptive_dropout_history % world_size != 0:
            raise AssertionError(f'--{continuous_concept_adaptive_dropout_history=} must be devisable by {world_size=}')

        self._continuous_concept_loss_history: deque[torch.Tensor] = deque(
            maxlen=continuous_concept_adaptive_dropout_history // world_size
        )
        self.world_size = world_size

    def forward(
            self,
            input_ids: torch.Tensor,
            labels: torch.Tensor | None = None,
            output_hidden_states: bool = False,
    ) -> SCTOutput:
        outputs = self.transformer(
            input_ids=input_ids,
            labels=None,
            output_hidden_states=(
                    output_hidden_states
                    or self.continuous_concept_mode != "none"
            ),
            return_dict=True,
            use_cache=False,
        )
        if labels is None:
            empty_loss = outputs.logits.sum() * 0.0
            return SCTOutput(
                loss=empty_loss,
                lm_loss=empty_loss,
                continuous_concept_loss=empty_loss,
                logits=outputs.logits,
                num_concepts=empty_loss,
                attention_allowed_density=empty_loss,
                hidden_states=outputs.hidden_states,
            )
        lm_loss = lm_loss_fn(outputs.logits, labels)
        continuous_concept_loss = torch.zeros((), dtype=lm_loss.dtype, device=lm_loss.device)
        if self.continuous_concept_mode != "none":
            hidden = select_continuous_concept_hidden_state(outputs.hidden_states, self.continuous_concept_layer)
            continuous_concept_loss = continuous_concept_loss_fn(
                hidden,
                self.concept_k,
                mode=self.continuous_concept_mode,
            )
            if self.continuous_concept_adaptive_dropout is not None:
                continuous_concept_loss = apply_continuous_concept_adaptive_dropout(
                    continuous_concept_loss,
                    self.continuous_concept_adaptive_dropout,
                    self._continuous_concept_loss_history,
                    self.training,
                    self.world_size,
                )
            else:
                continuous_concept_loss = apply_continuous_concept_dropout(
                    continuous_concept_loss, self.continuous_concept_dropout
                )

        loss = lm_loss + self.continuous_concept_lambda * continuous_concept_loss

        return SCTOutput(
            loss=loss,
            lm_loss=lm_loss,
            continuous_concept_loss=continuous_concept_loss,
            logits=outputs.logits,
            num_concepts=torch.zeros((), dtype=lm_loss.dtype, device=lm_loss.device),
            attention_allowed_density=torch.zeros((), dtype=lm_loss.dtype, device=lm_loss.device),
            hidden_states=outputs.hidden_states,
        )


class BottleneckGPT2LM(nn.Module):
    def __init__(
            self,
            transformer: GPT2LMHeadModel | GPTNeoXForCausalLM,
            bottleneck_mode: str,
            concept_k: int = 8,
            attention_allowed_density: float = 0.0,
            continuous_concept_mode: str = "none",
            continuous_concept_lambda: float = 0.01,
            continuous_concept_layer: str = "mid",
            continuous_concept_dropout: float | None = None,
            continuous_concept_adaptive_dropout: float | None = None,
            continuous_concept_adaptive_dropout_history: int = 100,
            world_size: int = 1
    ):
        super().__init__()
        self.transformer = transformer
        self.concept_k = concept_k
        self.bottleneck_mode = normalize_bottleneck_mode(bottleneck_mode)
        self.attention_allowed_density = attention_allowed_density
        self.continuous_concept_mode = normalize_continuous_concept_mode(continuous_concept_mode)
        self.continuous_concept_lambda = continuous_concept_lambda
        self.continuous_concept_layer = continuous_concept_layer
        self.continuous_concept_dropout = validate_continuous_concept_dropout(
            self.continuous_concept_mode, continuous_concept_dropout
        )
        self.continuous_concept_adaptive_dropout = validate_continuous_concept_adaptive_dropout(
            self.continuous_concept_mode,
            continuous_concept_adaptive_dropout,
            continuous_concept_adaptive_dropout_history,
            self.continuous_concept_dropout,
        )

        if continuous_concept_adaptive_dropout_history % world_size != 0:
            raise AssertionError(f'--{continuous_concept_adaptive_dropout_history=} must be devisable by {world_size=}')

        self._continuous_concept_loss_history: deque[torch.Tensor] = deque(
            maxlen=continuous_concept_adaptive_dropout_history // world_size
        )
        self.world_size = world_size

    def forward(
            self,
            input_ids: torch.Tensor,
            labels: torch.Tensor | None = None,
            output_hidden_states: bool = False,
    ) -> SCTOutput:
        outputs = self.transformer(
            input_ids=input_ids,
            labels=None,
            output_hidden_states=(
                    output_hidden_states
                    or self.continuous_concept_mode != "none"
            ),
            return_dict=True,
            use_cache=False,
        )
        if labels is None:
            labels = input_ids

        lm_loss = lm_loss_fn(outputs.logits, labels)
        continuous_concept_loss = torch.zeros((), dtype=lm_loss.dtype, device=lm_loss.device)
        if self.continuous_concept_mode != "none":
            hidden = select_continuous_concept_hidden_state(outputs.hidden_states, self.continuous_concept_layer)
            continuous_concept_loss = continuous_concept_loss_fn(
                hidden,
                self.concept_k,
                mode=self.continuous_concept_mode,
            )
            if self.continuous_concept_adaptive_dropout is not None:
                continuous_concept_loss = apply_continuous_concept_adaptive_dropout(
                    continuous_concept_loss,
                    self.continuous_concept_adaptive_dropout,
                    self._continuous_concept_loss_history,
                    self.training,
                    self.world_size
                )
            else:
                continuous_concept_loss = apply_continuous_concept_dropout(
                    continuous_concept_loss, self.continuous_concept_dropout
                )
        concept_mask = build_boundary_concept_mask(
            batch_size=input_ids.shape[0],
            seq_len=input_ids.shape[1],
            concept_k=self.concept_k,
            device=input_ids.device,
        )
        num_concepts = concept_mask.sum(dim=1)

        loss = lm_loss + self.continuous_concept_lambda * continuous_concept_loss

        return SCTOutput(
            loss=loss,
            lm_loss=lm_loss,
            continuous_concept_loss=continuous_concept_loss,
            logits=outputs.logits,
            num_concepts=num_concepts.float().mean(),
            attention_allowed_density=torch.tensor(
                self.attention_allowed_density,
                dtype=lm_loss.dtype,
                device=lm_loss.device,
            ),
            hidden_states=outputs.hidden_states,
        )


def normalize_continuous_concept_mode(mode: str) -> str:
    aliases = {
        "none": "none",
        "off": "none",
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
    if mode not in aliases:
        raise ValueError(
            f"Unknown continuous concept mode {mode!r}. "
            "Expected none, cosine, cosine_delta, norm, norm_l2, l2, mse, mse_r, mse_rs, mse_rsr, mse_nm, mse_nmp, mse2, mse_sg, or mse_rsg."
        )
    return aliases[mode]


def validate_continuous_concept_dropout(mode: str, dropout: float | None) -> float | None:
    if dropout is None:
        return None
    if mode != "mse":
        raise ValueError("--continuous_concept_dropout is only supported when --continuous_concept_mode is mse")
    if not 0.0 <= dropout <= 1.0:
        raise ValueError("--continuous_concept_dropout must be between 0.0 and 1.0")
    return dropout


def apply_continuous_concept_dropout(loss: torch.Tensor, dropout: float | None) -> torch.Tensor:
    """Drop the continuous-concept loss for the complete current batch."""
    if dropout is None or dropout == 0.0:
        return loss
    if dropout == 1.0 or torch.rand((), device=loss.device) < dropout:
        return loss * 0.0
    return loss


def validate_continuous_concept_adaptive_dropout(
        mode: str,
        percentile: float | None,
        history_length: int,
        fixed_dropout: float | None,
) -> float | None:
    if percentile is None:
        return None
    if mode != "mse":
        raise ValueError(
            "--continuous_concept_adaptive_dropout is only supported when --continuous_concept_mode is mse")
    if fixed_dropout is not None:
        raise ValueError(
            "--continuous_concept_adaptive_dropout and --continuous_concept_dropout are mutually exclusive")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("--continuous_concept_adaptive_dropout must be a percentile between 0 and 100")
    if history_length <= 0:
        raise ValueError("--continuous_concept_adaptive_dropout_history must be positive")
    return percentile


def gather_history(history: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size <= 1:
        return history

    tensors = [torch.zeros_like(history) for _ in range(world_size)]
    dist.all_gather(tensors, history)

    return torch.concat(tensors, dim=0)


def apply_continuous_concept_adaptive_dropout(
        loss: torch.Tensor,
        percentile: float | None,
        history: deque[torch.Tensor],
        training: bool,
        world_size: int = 1
) -> torch.Tensor:
    """Drop a training batch when its raw MSE CC loss is below its recent percentile."""
    if percentile is None or not training:
        return loss
    raw_loss = loss.detach().float()
    if len(history) < history.maxlen:
        history.append(raw_loss)
        return loss

    current_history = torch.stack(tuple(history)).to(device=loss.device)
    current_history = gather_history(current_history, world_size)

    threshold = torch.quantile(current_history, percentile / 100.0)
    history.append(raw_loss)
    return loss * (raw_loss >= threshold).to(dtype=loss.dtype)


def select_continuous_concept_hidden_state(hidden_states: tuple[torch.Tensor, ...], layer: str) -> torch.Tensor:
    if layer in {"mid", "middle"}:
        return hidden_states[len(hidden_states) // 2]
    if layer in {"last", "final"}:
        return hidden_states[-1]
    try:
        idx = int(layer)
    except ValueError as exc:
        raise ValueError("--continuous_concept_layer must be mid, last, or an integer index") from exc
    return hidden_states[idx]


def continuous_concept_loss_fn(hidden: torch.Tensor, concept_k: int = 8, mode: str = "none") -> torch.Tensor:
    mode = normalize_continuous_concept_mode(mode)
    if mode == "none":
        return torch.zeros((), dtype=hidden.dtype, device=hidden.device)
    if mode == "l2":
        # Dimension-normalized squared L2 norm, matching the scaling of mse.
        return hidden.pow(2).mean()
    if hidden.shape[1] < 2:
        return torch.zeros((), dtype=hidden.dtype, device=hidden.device)
    if mode == "cosine":
        return 1.0 - F.cosine_similarity(hidden[:, :-1, :], hidden[:, 1:, :], dim=-1).mean()
    if mode == "cosine_delta":
        if hidden.shape[1] < 3:
            return torch.zeros((), dtype=hidden.dtype, device=hidden.device)
        a = hidden[:, 1:-1, :] - hidden[:, :-2, :]
        b = hidden[:, 2:, :] - hidden[:, 1:-1, :]
        return 1.0 - F.cosine_similarity(a, b, dim=-1).mean()
    if mode in {"norm", "norm_l2"}:
        # Relative squared change normalized by the total energy of each pair.
        # Float32 keeps the stabilizing epsilon representable under fp16.
        hidden_float = hidden.float()
        h0 = hidden_float[:, :-1, :]
        h1 = hidden_float[:, 1:, :]
        diff2 = (h1 - h0).pow(2).sum(dim=-1)
        denom = h1.pow(2).sum(dim=-1) + h0.pow(2).sum(dim=-1) + 1e-6
        return (diff2 / denom).mean()
    if mode == "mse2":
        if hidden.shape[1] < 3:
            return torch.zeros((), dtype=hidden.dtype, device=hidden.device)
        return (2 * hidden[:, 2:, :] - hidden[:, 1:-1, :] - hidden[:, :-2, :]).pow(2).mean()
    if mode == "mse_r":
        if concept_k <= 0:
            raise ValueError("concept_k must be positive")
        # Consecutive boundary positions differ by concept_k. The denominator is
        # attached so minimizing the ratio repels their representations.
        hidden_float = hidden.float()
        boundary_hidden = hidden_float[:, concept_k - 1:: concept_k, :]
        if boundary_hidden.shape[1] < 2:
            return torch.zeros((), dtype=hidden.dtype, device=hidden.device)
        step_mse = (hidden_float[:, 1:, :] - hidden_float[:, :-1, :]).pow(2).mean()
        boundary_mse = (boundary_hidden[:, 1:, :] - boundary_hidden[:, :-1, :]).pow(2).mean()
        return step_mse / (boundary_mse + 1e-8)
    if mode == "mse_rs":
        if concept_k <= 0:
            raise ValueError("concept_k must be positive")
        # Squaring the boundary MSE makes the repulsive effect decay faster as
        # boundary representations separate.
        hidden_float = hidden.float()
        boundary_hidden = hidden_float[:, concept_k - 1:: concept_k, :]
        if boundary_hidden.shape[1] < 2:
            return torch.zeros((), dtype=hidden.dtype, device=hidden.device)
        step_mse = (hidden_float[:, 1:, :] - hidden_float[:, :-1, :]).pow(2).mean()
        boundary_mse = (boundary_hidden[:, 1:, :] - boundary_hidden[:, :-1, :]).pow(2).mean()
        return step_mse / (boundary_mse.pow(2) + 1e-8)
    if mode == "mse_rsr":
        if concept_k <= 0:
            raise ValueError("concept_k must be positive")
        # Use RMS boundary separation rather than squared boundary separation.
        hidden_float = hidden.float()
        boundary_hidden = hidden_float[:, concept_k - 1:: concept_k, :]
        if boundary_hidden.shape[1] < 2:
            return torch.zeros((), dtype=hidden.dtype, device=hidden.device)
        step_mse = (hidden_float[:, 1:, :] - hidden_float[:, :-1, :]).pow(2).mean()
        boundary_rms = (boundary_hidden[:, 1:, :] - boundary_hidden[:, :-1, :]).pow(2).mean().sqrt()
        return step_mse / (boundary_rms + 1e-8)
    if mode == "mse_nm":
        # Normalize each token's squared change by its own squared L2 norm.
        # Float32 keeps the stabilizing epsilon representable under fp16.
        hidden_float = hidden.float()
        delta_sq = (hidden_float[:, 1:, :] - hidden_float[:, :-1, :]).pow(2).sum(dim=-1)
        hidden_sq = hidden_float[:, 1:, :].pow(2).sum(dim=-1)
        return (delta_sq / (hidden_sq + 1e-8)).mean()
    if mode == "mse_nmp":
        # Normalize pooled squared change by pooled squared hidden magnitude.
        # Float32 keeps the stabilizing epsilon representable under fp16.
        hidden_float = hidden.float()
        delta_mse = (hidden_float[:, 1:, :] - hidden_float[:, :-1, :]).pow(2).mean()
        hidden_mse = hidden_float[:, 1:, :].pow(2).mean()
        return delta_mse / (hidden_mse + 1e-8)
    if mode in {"mse", "mse_sg", "mse_rsg"}:
        if mode != "mse":
            if concept_k <= 0:
                raise ValueError("concept_k must be positive")
            pos = torch.arange(hidden.shape[1], device=hidden.device)
            boundary = ((pos % concept_k) == (concept_k - 1)).view(1, -1, 1)
            detach_mask = boundary if mode == "mse_sg" else ~boundary
            hidden = torch.where(detach_mask, hidden.detach(), hidden)
        return (hidden[:, 1:, :] - hidden[:, :-1, :]).pow(2).mean()
    raise RuntimeError(f"Unhandled continuous concept mode: {mode}")


def build_gpt2_config(train_config: TrainConfig, vocab_size: int, eos_token_id: int | None = None) -> GPT2Config:
    dims = model_dims(train_config)
    n_positions = train_config.n_positions or train_config.seq_len

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_ctx=n_positions,
        n_layer=dims["n_layer"],
        n_head=dims["n_head"],
        n_embd=dims["n_embd"],
        resid_pdrop=train_config.dropout,
        embd_pdrop=train_config.dropout,
        attn_pdrop=train_config.dropout,
        bos_token_id=eos_token_id,
        eos_token_id=eos_token_id,
    )

    config._attn_implementation = "eager"
    return config


def build_model(
        train_config: TrainConfig,
        vocab_size: int,
        eos_token_id: int,
        world_size: int = 1
) -> nn.Module:
    bottleneck_mode = normalize_bottleneck_mode(train_config.bottleneck_mode)

    if train_config.init_from_pretrained:
        if train_config.model_size != "gpt2-small":
            raise ValueError("--init_from_pretrained currently supports --model_size gpt2-small only")
        transformer = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
        transformer.config._attn_implementation = "eager"
        if transformer.config.vocab_size != vocab_size:
            transformer.resize_token_embeddings(vocab_size)
        required_positions = train_config.n_positions or train_config.seq_len
        if transformer.config.n_positions < required_positions:
            raise ValueError("Pretrained GPT-2 positional embeddings are too short for this configuration")
    elif train_config.model_size.lower().startswith('gpt2'):
        transformer = GPT2LMHeadModel(build_gpt2_config(train_config, vocab_size, eos_token_id))
    elif train_config.model_size.lower().startswith('pythia'):
        model_config = GPTNeoXConfig.from_pretrained(f'EleutherAI/{train_config.model_size}')
        model_config._attn_implementation = 'eager'
        transformer = GPTNeoXForCausalLM(model_config)
    else:
        raise ValueError(f'Unrecognized model {train_config.model_size}')

    if bottleneck_mode is not None:
        is_gpt2 = train_config.model_size.lower().startswith('gpt2')
        max_position_embeddings = transformer.config.n_positions if is_gpt2 else transformer.config.max_position_embeddings

        if max_position_embeddings < train_config.seq_len:
            raise ValueError("Model positional length is shorter than sequence length")

        attention_allowed_density = patch_bottleneck_attention(
            transformer,
            train_config.seq_len,
            train_config.bottleneck_concept_k,
            bottleneck_mode,
            train_config.bottleneck_layers,
            train_config.seed,
        )

        if train_config.change_rope_layers is not None:
            assert not isinstance(transformer, GPT2LMHeadModel), 'GPT2 models have no RoPE, set change_rope_layers = None'
            add_rope_changing_hook(transformer, train_config)

        return BottleneckGPT2LM(
            transformer=transformer,
            concept_k=train_config.bottleneck_concept_k,
            bottleneck_mode=bottleneck_mode,
            attention_allowed_density=attention_allowed_density,
            continuous_concept_mode=train_config.continuous_concept_mode,
            continuous_concept_lambda=train_config.continuous_concept_lambda,
            continuous_concept_layer=train_config.continuous_concept_layer,
            continuous_concept_dropout=train_config.continuous_concept_dropout,
            continuous_concept_adaptive_dropout=train_config.continuous_concept_adaptive_dropout,
            continuous_concept_adaptive_dropout_history=train_config.continuous_concept_adaptive_dropout_history,
            world_size=world_size
        )

    return VanillaGPT2LM(
        transformer,
        concept_k=train_config.bottleneck_concept_k,
        continuous_concept_mode=train_config.continuous_concept_mode,
        continuous_concept_lambda=train_config.continuous_concept_lambda,
        continuous_concept_layer=train_config.continuous_concept_layer,
        continuous_concept_dropout=train_config.continuous_concept_dropout,
        continuous_concept_adaptive_dropout=train_config.continuous_concept_adaptive_dropout,
        continuous_concept_adaptive_dropout_history=train_config.continuous_concept_adaptive_dropout_history,
        world_size=world_size
    )


def _test_bottleneck_mask() -> None:
    boundary = build_bottleneck_allow_mask(seq_len=8, concept_k=4, device=torch.device("cpu"), mode="boundary")
    assert bool(boundary[0, 0])
    assert not bool(boundary[0, 1])
    assert bool(boundary[3, 0])
    assert bool(boundary[3, 3])
    assert bool(boundary[4, 3])
    assert not bool(boundary[4, 0])
    assert not bool(boundary[4, 1])
    assert not bool(boundary[4, 2])
    assert bool(boundary[6, 3])
    assert bool(boundary[6, 4])
    assert bool(boundary[6, 5])
    assert bool(boundary[6, 6])
    assert int(boundary.sum(dim=-1).min().item()) > 0

    sliding = build_bottleneck_allow_mask(seq_len=8, concept_k=4, device=torch.device("cpu"), mode="sliding_window")
    assert int(sliding.sum(dim=-1).max().item()) == int(boundary.sum(dim=-1).max().item())
    assert bool(sliding[4, 3])
    assert bool(sliding[4, 4])
    assert not bool(sliding[4, 2])

    random_boundary = build_bottleneck_allow_mask(seq_len=8, concept_k=4, device=torch.device("cpu"),
                                                  mode="random_boundary", seed=82)
    assert int(random_boundary.sum(dim=-1).min().item()) > 0
    assert torch.equal(
        random_boundary,
        build_bottleneck_allow_mask(seq_len=8, concept_k=4, device=torch.device("cpu"), mode="random_boundary",
                                    seed=82),
    )

    random_sparse = build_bottleneck_allow_mask(seq_len=8, concept_k=4, device=torch.device("cpu"),
                                                mode="random_sparse", seed=82)
    assert torch.equal(random_sparse.sum(dim=-1), boundary.sum(dim=-1))
    assert torch.equal(
        random_sparse,
        build_bottleneck_allow_mask(seq_len=8, concept_k=4, device=torch.device("cpu"), mode="random_sparse", seed=82),
    )
    print("bottleneck mask sanity checks passed")


def _test_continuous_concept_loss() -> None:
    dropout_loss = torch.tensor(1.0, requires_grad=True)
    assert apply_continuous_concept_dropout(dropout_loss, None) is dropout_loss
    assert apply_continuous_concept_dropout(dropout_loss, 0.0) is dropout_loss
    dropped_loss = apply_continuous_concept_dropout(dropout_loss, 1.0)
    assert dropped_loss.item() == 0.0
    dropped_loss.backward()
    assert dropout_loss.grad is not None and dropout_loss.grad.item() == 0.0
    assert validate_continuous_concept_dropout("mse", 0.5) == 0.5
    try:
        validate_continuous_concept_dropout("norm", 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("continuous concept dropout must require mse mode")

    adaptive_history: deque[torch.Tensor] = deque(maxlen=3)
    for value in (1.0, 2.0, 3.0):
        adaptive_loss = torch.tensor(value, requires_grad=True)
        assert apply_continuous_concept_adaptive_dropout(adaptive_loss, 50.0, adaptive_history, True) is adaptive_loss
    dropped_adaptive_loss = apply_continuous_concept_adaptive_dropout(
        torch.tensor(0.5, requires_grad=True), 50.0, adaptive_history, True
    )
    assert dropped_adaptive_loss.item() == 0.0
    history_size = len(adaptive_history)
    evaluation_loss = torch.tensor(0.25, requires_grad=True)
    assert apply_continuous_concept_adaptive_dropout(evaluation_loss, 50.0, adaptive_history, False) is evaluation_loss
    assert len(adaptive_history) == history_size
    assert validate_continuous_concept_adaptive_dropout("mse", 50.0, 100, None) == 50.0
    try:
        validate_continuous_concept_adaptive_dropout("mse", 50.0, 100, 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("adaptive and fixed continuous concept dropout must be mutually exclusive")

    base = torch.tensor([0.0, 1.0, 4.0, 9.0, 16.0, 25.0]).view(1, 6, 1)
    concept_k = 3
    boundary_positions = torch.tensor([2, 5])
    interior_positions = torch.tensor([0, 1, 3, 4])

    hidden_sg = base.clone().requires_grad_(True)
    continuous_concept_loss_fn(hidden_sg, concept_k, mode="mse_sg").backward()
    grad_sg = hidden_sg.grad.view(-1)
    assert torch.allclose(grad_sg[boundary_positions], torch.zeros_like(grad_sg[boundary_positions]))
    assert bool((grad_sg[interior_positions].abs() > 0).all())

    hidden_rsg = base.clone().requires_grad_(True)
    continuous_concept_loss_fn(hidden_rsg, concept_k, mode="mse_rsg").backward()
    grad_rsg = hidden_rsg.grad.view(-1)
    assert torch.allclose(grad_rsg[interior_positions], torch.zeros_like(grad_rsg[interior_positions]))
    assert bool((grad_rsg[boundary_positions].abs() > 0).all())

    mse2_loss = continuous_concept_loss_fn(base, mode="mse2")
    expected_mse2_loss = (2 * base[:, 2:, :] - base[:, 1:-1, :] - base[:, :-2, :]).pow(2).mean()
    assert torch.allclose(mse2_loss, expected_mse2_loss)
    assert torch.equal(continuous_concept_loss_fn(base[:, :2, :], mode="mse2"), torch.zeros(()))

    norm_loss = continuous_concept_loss_fn(base, mode="norm")
    h0, h1 = base[:, :-1, :], base[:, 1:, :]
    expected_norm_loss = ((h1 - h0).pow(2).sum(dim=-1) / (h1.pow(2).sum(dim=-1) + h0.pow(2).sum(dim=-1) + 1e-6)).mean()
    assert torch.allclose(norm_loss, expected_norm_loss)
    assert torch.allclose(continuous_concept_loss_fn(base, mode="norm_l2"), expected_norm_loss)

    mse_r_loss = continuous_concept_loss_fn(base, concept_k, mode="mse_r")
    expected_mse_r_loss = (
            (base[:, 1:, :] - base[:, :-1, :]).pow(2).mean()
            / ((base[:, [5], :] - base[:, [2], :]).pow(2).mean() + 1e-8)
    )
    assert torch.allclose(mse_r_loss, expected_mse_r_loss)
    hidden_r = base.clone().requires_grad_(True)
    continuous_concept_loss_fn(hidden_r, concept_k, mode="mse_r").backward()
    assert bool((hidden_r.grad[:, boundary_positions, :].abs() > 0).all())

    mse_rs_loss = continuous_concept_loss_fn(base, concept_k, mode="mse_rs")
    expected_mse_rs_loss = (
            (base[:, 1:, :] - base[:, :-1, :]).pow(2).mean()
            / ((base[:, [5], :] - base[:, [2], :]).pow(2).mean().pow(2) + 1e-8)
    )
    assert torch.allclose(mse_rs_loss, expected_mse_rs_loss)

    mse_rsr_loss = continuous_concept_loss_fn(base, concept_k, mode="mse_rsr")
    expected_mse_rsr_loss = (
            (base[:, 1:, :] - base[:, :-1, :]).pow(2).mean()
            / ((base[:, [5], :] - base[:, [2], :]).pow(2).mean().sqrt() + 1e-8)
    )
    assert torch.allclose(mse_rsr_loss, expected_mse_rsr_loss)

    l2_loss = continuous_concept_loss_fn(base, mode="l2")
    assert torch.allclose(l2_loss, base.pow(2).mean())

    mse_nm_loss = continuous_concept_loss_fn(base, mode="mse_nm")
    expected_mse_nm_loss = (
            (base[:, 1:, :] - base[:, :-1, :]).pow(2).sum(dim=-1)
            / (base[:, 1:, :].pow(2).sum(dim=-1) + 1e-8)
    ).mean()
    assert torch.allclose(mse_nm_loss, expected_mse_nm_loss)

    mse_nmp_loss = continuous_concept_loss_fn(base, mode="mse_nmp")
    expected_mse_nmp_loss = (
            (base[:, 1:, :] - base[:, :-1, :]).pow(2).mean()
            / (base[:, 1:, :].pow(2).mean() + 1e-8)
    )
    assert torch.allclose(mse_nmp_loss, expected_mse_nmp_loss)

    hidden_with_sg_loss = base.clone().requires_grad_(True)
    cc_loss = continuous_concept_loss_fn(hidden_with_sg_loss, mode="mse")
    (0.25 * cc_loss + 0.75 * cc_loss.detach()).backward()
    hidden_gradient_with_sg_loss = hidden_with_sg_loss.grad.clone()
    hidden_gradient_without_sg_loss = base.clone().requires_grad_(True)
    (0.25 * continuous_concept_loss_fn(hidden_gradient_without_sg_loss, mode="mse")).backward()
    assert torch.allclose(hidden_gradient_with_sg_loss, hidden_gradient_without_sg_loss.grad)

    print("continuous concept loss sanity checks passed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_bottleneck_mask", action="store_true")
    parser.add_argument("--test_continuous_concept_loss", action="store_true")
    args = parser.parse_args()
    if args.test_bottleneck_mask:
        _test_bottleneck_mask()
    if args.test_continuous_concept_loss:
        _test_continuous_concept_loss()

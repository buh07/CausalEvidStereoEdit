from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.pytorch_utils import Conv1D

from stereacl.modeling import locate_attn_module, locate_decoder_blocks


@dataclass(frozen=True)
class AttentionProjectionSpec:
    layer_idx: int
    projection_name: str
    projection_kind: str
    in_features: int
    out_features: int
    num_heads: int
    head_dim: int
    projection_module: nn.Module


def locate_attention_output_projection(attn_module: nn.Module) -> tuple[str, nn.Module]:
    for name in ("o_proj", "c_proj", "out_proj", "dense"):
        if hasattr(attn_module, name):
            module = getattr(attn_module, name)
            if isinstance(module, (nn.Linear, Conv1D)):
                return name, module
    raise ValueError(f"Could not locate supported attention output projection in {type(attn_module).__name__}")


def _projection_io_features(proj: nn.Module) -> tuple[str, int, int]:
    if isinstance(proj, nn.Linear):
        return "linear", int(proj.in_features), int(proj.out_features)
    if isinstance(proj, Conv1D):
        # Conv1D stores weight as [in_features, out_features].
        in_features = int(proj.weight.shape[0])
        out_features = int(proj.weight.shape[1])
        return "conv1d", in_features, out_features
    raise TypeError(f"Unsupported projection module type: {type(proj).__name__}")


def _infer_num_heads_and_head_dim(attn_module: nn.Module, proj_in_features: int) -> tuple[int, int]:
    head_dim_candidates = [
        getattr(attn_module, "head_dim", None),
    ]
    num_head_candidates = [
        getattr(attn_module, "num_heads", None),
        getattr(attn_module, "num_attention_heads", None),
        getattr(attn_module, "n_head", None),
        getattr(attn_module, "total_num_heads", None),
    ]

    head_dim: int | None = None
    for value in head_dim_candidates:
        if isinstance(value, int) and value > 0:
            head_dim = int(value)
            break
    num_heads: int | None = None
    for value in num_head_candidates:
        if isinstance(value, int) and value > 0:
            num_heads = int(value)
            break

    if num_heads is not None and head_dim is None:
        if proj_in_features % num_heads != 0:
            raise ValueError(
                f"Projection input dim {proj_in_features} is not divisible by num_heads={num_heads}."
            )
        head_dim = proj_in_features // num_heads
    elif head_dim is not None and num_heads is None:
        if proj_in_features % head_dim != 0:
            raise ValueError(
                f"Projection input dim {proj_in_features} is not divisible by head_dim={head_dim}."
            )
        num_heads = proj_in_features // head_dim
    elif num_heads is None and head_dim is None:
        raise ValueError(
            f"Cannot infer attention geometry for module {type(attn_module).__name__}; "
            "missing head_dim/num_heads attributes."
        )

    assert num_heads is not None and head_dim is not None
    if num_heads * head_dim != proj_in_features:
        raise ValueError(
            "Inferred attention geometry mismatch: "
            f"num_heads={num_heads}, head_dim={head_dim}, proj_in_features={proj_in_features}"
        )
    return num_heads, head_dim


def build_attention_projection_specs(model: PreTrainedModel) -> dict[int, AttentionProjectionSpec]:
    specs: dict[int, AttentionProjectionSpec] = {}
    blocks = locate_decoder_blocks(model)
    for layer_idx, block in enumerate(blocks):
        attn_module = locate_attn_module(block)
        projection_name, projection_module = locate_attention_output_projection(attn_module)
        proj_kind, in_features, out_features = _projection_io_features(projection_module)
        num_heads, head_dim = _infer_num_heads_and_head_dim(attn_module, in_features)
        specs[layer_idx] = AttentionProjectionSpec(
            layer_idx=layer_idx,
            projection_name=projection_name,
            projection_kind=proj_kind,
            in_features=in_features,
            out_features=out_features,
            num_heads=num_heads,
            head_dim=head_dim,
            projection_module=projection_module,
        )
    return specs


def _reshape_to_heads(preproj_input: torch.Tensor, spec: AttentionProjectionSpec) -> torch.Tensor:
    if preproj_input.shape[-1] != spec.in_features:
        raise ValueError(
            f"Unexpected pre-projection width {preproj_input.shape[-1]} for layer {spec.layer_idx}; "
            f"expected {spec.in_features}"
        )
    target_shape = (*preproj_input.shape[:-1], spec.num_heads, spec.head_dim)
    return preproj_input.view(target_shape)


def attention_head_writes_from_preproj(
    preproj_input: torch.Tensor,
    spec: AttentionProjectionSpec,
) -> torch.Tensor:
    """
    Compute exact per-head residual writes before residual addition.

    Returns a tensor with shape [batch, seq, num_heads, out_features].
    Bias terms are intentionally excluded so writes sum to the linear projected
    attention output without duplicated bias per head.
    """
    head_states = _reshape_to_heads(preproj_input, spec)  # [..., H, D]
    if spec.projection_kind == "linear":
        proj = spec.projection_module
        assert isinstance(proj, nn.Linear)
        weight = proj.weight  # [out, in]
        reshaped_weight = weight.view(spec.out_features, spec.num_heads, spec.head_dim)
        # [..., h, d] x [o, h, d] -> [..., h, o]
        return torch.einsum("...hd,ohd->...ho", head_states, reshaped_weight)
    if spec.projection_kind == "conv1d":
        proj = spec.projection_module
        assert isinstance(proj, Conv1D)
        weight = proj.weight  # [in, out]
        reshaped_weight = weight.view(spec.num_heads, spec.head_dim, spec.out_features)
        # [..., h, d] x [h, d, o] -> [..., h, o]
        return torch.einsum("...hd,hdo->...ho", head_states, reshaped_weight)
    raise TypeError(f"Unsupported projection kind: {spec.projection_kind}")


def make_attention_head_replace_hook(
    spec: AttentionProjectionSpec,
    position: int,
    head_index: int,
    replacement_head_vector: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if head_index < 0 or head_index >= spec.num_heads:
        raise ValueError(f"head_index {head_index} is out of range for {spec.num_heads} heads.")

    def _hook(preproj_input: torch.Tensor) -> torch.Tensor:
        patched = preproj_input.clone()
        if position >= patched.shape[1]:
            return patched
        start = head_index * spec.head_dim
        end = start + spec.head_dim
        patched[:, position, start:end] = replacement_head_vector
        return patched

    return _hook


def make_attention_head_zero_hook(
    spec: AttentionProjectionSpec,
    position: int,
    head_index: int,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if head_index < 0 or head_index >= spec.num_heads:
        raise ValueError(f"head_index {head_index} is out of range for {spec.num_heads} heads.")

    def _hook(preproj_input: torch.Tensor) -> torch.Tensor:
        patched = preproj_input.clone()
        if position >= patched.shape[1]:
            return patched
        start = head_index * spec.head_dim
        end = start + spec.head_dim
        patched[:, position, start:end] = 0.0
        return patched

    return _hook


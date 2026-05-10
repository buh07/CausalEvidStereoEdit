from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _parse_device_map_spec(device: str) -> str | None:
    """
    Parse special multi-GPU device specs from the existing --device flag.

    Supported values:
    - shard-auto
    - multi-auto
    - device_map:auto
    - device_map:<strategy>
    """
    if device in {"shard-auto", "multi-auto", "device_map:auto"}:
        return "auto"
    if device.startswith("device_map:"):
        strategy = device.split(":", 1)[1].strip()
        return strategy or "auto"
    return None


def _infer_primary_device(model: PreTrainedModel, fallback: torch.device) -> torch.device:
    """
    Infer a reasonable input device for sharded models loaded with HF device_map.
    """
    hf_map = getattr(model, "hf_device_map", None)
    if not isinstance(hf_map, dict) or not hf_map:
        return fallback

    # Keep deterministic order from hf_device_map insertion order.
    for value in hf_map.values():
        if isinstance(value, int):
            return torch.device(f"cuda:{value}")
        if isinstance(value, str) and value.startswith("cuda:"):
            return torch.device(value)
    return fallback


@dataclass(frozen=True)
class ModelBundle:
    model_name: str
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    device: torch.device

    @property
    def num_layers(self) -> int:
        return len(locate_decoder_blocks(self.model))


def load_model_bundle(
    model_name: str,
    device: str = "auto",
    torch_dtype: str = "auto",
) -> ModelBundle:
    device_map = _parse_device_map_spec(device)
    if device_map is None:
        env_device_map = os.environ.get("STEREACL_DEVICE_MAP", "").strip()
        if env_device_map:
            device_map = env_device_map

    if device_map is not None:
        target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        target_device = resolve_device(device)
    dtype = None
    if torch_dtype != "auto":
        if not hasattr(torch, torch_dtype):
            raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
        dtype = getattr(torch, torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {}
    if dtype is not None:
        model_kwargs["dtype"] = dtype

    if device_map is not None:
        if not torch.cuda.is_available():
            raise ValueError("device_map requested but CUDA is not available.")
        model_kwargs["device_map"] = device_map
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        target_device = _infer_primary_device(model, torch.device("cuda:0"))
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.to(target_device)

    model.eval()
    return ModelBundle(
        model_name=model_name,
        tokenizer=tokenizer,
        model=model,
        device=target_device,
    )


def locate_decoder_blocks(model: PreTrainedModel) -> list[nn.Module]:
    candidates = [
        "model.layers",  # Llama/Gemma/Mistral
        "transformer.h",  # GPT-2
        "gpt_neox.layers",  # GPT-NeoX
        "model.decoder.layers",  # OPT/BART-like decoder
    ]
    for path in candidates:
        obj: object = model
        found = True
        for part in path.split("."):
            if not hasattr(obj, part):
                found = False
                break
            obj = getattr(obj, part)
        if found and isinstance(obj, nn.ModuleList) and len(obj) > 0:
            return list(obj)
    raise ValueError("Could not locate decoder blocks for this model architecture.")


def locate_attn_module(block: nn.Module) -> nn.Module:
    for name in ("self_attn", "attn", "attention"):
        if hasattr(block, name):
            return getattr(block, name)
    raise ValueError(f"Could not locate attention module in block type {type(block).__name__}")


def locate_mlp_module(block: nn.Module) -> nn.Module:
    for name in ("mlp", "feed_forward", "ffn"):
        if hasattr(block, name):
            return getattr(block, name)
    raise ValueError(f"Could not locate MLP module in block type {type(block).__name__}")


def extract_unembedding_matrix(model: PreTrainedModel) -> torch.Tensor:
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        # Shape: [vocab, d_model], transpose to [d_model, vocab]
        return model.lm_head.weight.detach().T
    raise ValueError("Could not locate LM head weight for unembedding extraction.")


def encode_text(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    device: torch.device,
    max_length: int = 256,
) -> dict[str, torch.Tensor]:
    batch = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    return {k: v.to(device) for k, v in batch.items()}


@dataclass
class ForwardCapture:
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]
    attention_outputs: dict[int, torch.Tensor]
    mlp_outputs: dict[int, torch.Tensor]


def forward_with_component_capture(
    model: PreTrainedModel,
    encoded_inputs: dict[str, torch.Tensor],
    require_grad: bool = False,
    output_hidden_states: bool = True,
    capture_attention: bool = True,
    capture_mlp: bool = True,
    retain_grad_on_captures: bool = False,
    attention_patch_map: dict[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    mlp_patch_map: dict[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    residual_patch_map: dict[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
) -> ForwardCapture:
    blocks = locate_decoder_blocks(model)
    attn_outputs: dict[int, torch.Tensor] = {}
    mlp_outputs: dict[int, torch.Tensor] = {}
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def _normalize_hook_output(out: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if isinstance(out, tuple):
            return out[0]
        return out

    # Residual-stream hooks: applied as pre-hooks on each block so the full
    # accumulated residual (h_{l-1}) is intercepted before attention+MLP add
    # their writes. This is the correct place to project out a direction from
    # the complete residual stream (cf. Arditi et al. 2024).
    if residual_patch_map:
        for layer_idx, block in enumerate(blocks):
            if layer_idx not in residual_patch_map:
                continue

            def _make_residual_pre_hook(idx: int) -> Callable:
                def _hook(_module: nn.Module, args: tuple) -> tuple | None:
                    if not args or not isinstance(args[0], torch.Tensor):
                        return None
                    patched = residual_patch_map[idx](args[0])
                    return (patched,) + args[1:]
                return _hook

            hooks.append(block.register_forward_pre_hook(_make_residual_pre_hook(layer_idx)))

    for layer_idx, block in enumerate(blocks):
        if capture_attention:
            attn_module = locate_attn_module(block)

            def _make_attn_hook(idx: int) -> Callable:
                def _hook(_module: nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor | tuple[torch.Tensor, ...]):
                    out_tensor = _normalize_hook_output(out)
                    patched = (
                        attention_patch_map[idx](out_tensor)
                        if attention_patch_map and idx in attention_patch_map
                        else out_tensor
                    )
                    if retain_grad_on_captures and patched.requires_grad:
                        patched.retain_grad()
                    attn_outputs[idx] = patched
                    if isinstance(out, tuple):
                        out_list = list(out)
                        out_list[0] = patched
                        return tuple(out_list)
                    return patched

                return _hook

            hooks.append(attn_module.register_forward_hook(_make_attn_hook(layer_idx)))

        if capture_mlp:
            mlp_module = locate_mlp_module(block)

            def _make_mlp_hook(idx: int) -> Callable:
                def _hook(_module: nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor | tuple[torch.Tensor, ...]):
                    out_tensor = _normalize_hook_output(out)
                    patched = (
                        mlp_patch_map[idx](out_tensor)
                        if mlp_patch_map and idx in mlp_patch_map
                        else out_tensor
                    )
                    if retain_grad_on_captures and patched.requires_grad:
                        patched.retain_grad()
                    mlp_outputs[idx] = patched
                    if isinstance(out, tuple):
                        out_list = list(out)
                        out_list[0] = patched
                        return tuple(out_list)
                    return patched

                return _hook

            hooks.append(mlp_module.register_forward_hook(_make_mlp_hook(layer_idx)))

    try:
        with torch.set_grad_enabled(require_grad):
            outputs = model(
                **encoded_inputs,
                output_hidden_states=output_hidden_states,
                use_cache=False,
            )
    finally:
        for handle in hooks:
            handle.remove()

    hidden_states: tuple[torch.Tensor, ...]
    if output_hidden_states and outputs.hidden_states is not None:
        hidden_states = tuple(outputs.hidden_states)
    else:
        hidden_states = tuple()

    return ForwardCapture(
        logits=outputs.logits,
        hidden_states=hidden_states,
        attention_outputs=attn_outputs,
        mlp_outputs=mlp_outputs,
    )

from __future__ import annotations

from collections.abc import Callable

import torch


def make_direction_projection_hook(direction: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    d_base = direction.detach().float().cpu()
    norm_sq_base = float(torch.dot(d_base, d_base).item())
    if norm_sq_base == 0.0:
        return lambda x: x

    per_device: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _hook(out: torch.Tensor) -> torch.Tensor:
        original_dtype = out.dtype
        x = out.float()
        dev_key = str(x.device)
        if dev_key not in per_device:
            d_dev = d_base.to(device=x.device, dtype=torch.float32)
            norm_sq_dev = torch.dot(d_dev, d_dev).clamp(min=1e-12)
            per_device[dev_key] = (d_dev, norm_sq_dev)
        d_dev, norm_sq_dev = per_device[dev_key]
        proj_coeff = (x @ d_dev) / norm_sq_dev
        projected = x - proj_coeff.unsqueeze(-1) * d_dev.unsqueeze(0).unsqueeze(0)
        return projected.to(original_dtype)

    return _hook


def make_zero_position_hook(position: int) -> Callable[[torch.Tensor], torch.Tensor]:
    def _hook(out: torch.Tensor) -> torch.Tensor:
        patched = out.clone()
        if position < patched.shape[1]:
            patched[:, position, :] = 0.0
        return patched

    return _hook


def make_replace_position_hook(
    position: int,
    replacement_vector: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    repl_base = replacement_vector.detach().float().cpu()
    per_device: dict[str, torch.Tensor] = {}

    def _hook(out: torch.Tensor) -> torch.Tensor:
        patched = out.clone()
        if position < patched.shape[1]:
            dev_key = str(patched.device)
            if dev_key not in per_device:
                per_device[dev_key] = repl_base.to(device=patched.device, dtype=patched.dtype)
            patched[:, position, :] = per_device[dev_key]
        return patched

    return _hook


def make_direction_injection_hook(
    direction: torch.Tensor,
    alpha: float,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Add alpha * d̂ to every token position in the output tensor."""
    d_base = direction.detach().float().cpu()
    norm = d_base.norm()
    d_hat_base = d_base / norm.clamp(min=1e-8)
    per_device: dict[str, torch.Tensor] = {}

    def _hook(out: torch.Tensor) -> torch.Tensor:
        orig_dtype = out.dtype
        dev_key = str(out.device)
        if dev_key not in per_device:
            per_device[dev_key] = d_hat_base.to(device=out.device, dtype=torch.float32)
        return (out.float() + alpha * per_device[dev_key]).to(orig_dtype)

    return _hook


def make_direction_injection_at_position_hook(
    position: int,
    direction: torch.Tensor,
    alpha: float,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Add alpha * d̂ to one token position in the residual stream."""
    d_base = direction.detach().float().cpu()
    norm = d_base.norm()
    d_hat_base = d_base / norm.clamp(min=1e-8)
    per_device: dict[str, torch.Tensor] = {}

    def _hook(out: torch.Tensor) -> torch.Tensor:
        if position >= out.shape[1]:
            return out
        orig_dtype = out.dtype
        patched = out.float().clone()
        dev_key = str(out.device)
        if dev_key not in per_device:
            per_device[dev_key] = d_hat_base.to(device=out.device, dtype=torch.float32)
        patched[:, position, :] = patched[:, position, :] + (alpha * per_device[dev_key])
        return patched.to(orig_dtype)

    return _hook


def make_rank_k_projection_hook(
    direction_matrix: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Project out the subspace spanned by columns of direction_matrix.

    direction_matrix: shape (d_model, k) — each column is a direction vector.
    Columns are QR-orthonormalized before projecting.
    """
    q_base, _ = torch.linalg.qr(direction_matrix.detach().float().cpu())  # Q: (d_model, k)
    per_device: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _hook(out: torch.Tensor) -> torch.Tensor:
        orig_dtype = out.dtype
        x = out.float()
        dev_key = str(out.device)
        if dev_key not in per_device:
            q_dev = q_base.to(device=out.device, dtype=torch.float32)
            per_device[dev_key] = (q_dev, q_dev.T)
        q_dev, qt_dev = per_device[dev_key]
        x = x - (x @ q_dev) @ qt_dev  # remove span: x - Q(Q^T x)
        return x.to(orig_dtype)

    return _hook


def make_direction_projection_at_position_hook(
    position: int,
    direction: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Project out the stereotype direction from only one token position in the residual stream.

    Unlike make_direction_projection_hook (which projects from all positions), this hook
    modifies only hidden_states[:, position, :], leaving all other positions unchanged.
    Used to test whether stereotype information at a specific position (e.g. prediction_position)
    is causally upstream of the final stereotype logit.
    """
    d_base = direction.detach().float().cpu()
    if float(torch.dot(d_base, d_base).item()) == 0.0:
        return lambda x: x
    per_device: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _hook(out: torch.Tensor) -> torch.Tensor:
        original_dtype = out.dtype
        x = out.float()
        if position >= x.shape[1]:
            return out
        dev_key = str(x.device)
        if dev_key not in per_device:
            d_dev = d_base.to(device=x.device, dtype=torch.float32)
            norm_sq_dev = torch.dot(d_dev, d_dev).clamp(min=1e-12)
            per_device[dev_key] = (d_dev, norm_sq_dev)
        d_dev, norm_sq_dev = per_device[dev_key]
        h_pos = x[:, position, :]  # (batch, d_model)
        proj_coeff = (h_pos @ d_dev) / norm_sq_dev  # (batch,)
        projected = x.clone()
        projected[:, position, :] = h_pos - proj_coeff.unsqueeze(-1) * d_dev
        return projected.to(original_dtype)

    return _hook

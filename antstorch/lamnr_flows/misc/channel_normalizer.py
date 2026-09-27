"""Per-channel intensity normalizer shared by the LAMNr trainers.

Used by the hybrid trainer (``signal1d`` views and image views with
``intensity="0mean"``) and by the Glow 2D/3D trainers (``--intensity 0mean``).
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch

__all__ = ["ChannelNormalizer"]


class ChannelNormalizer:
    """Per-channel scaler for ``(channels, *spatial_or_time)`` tensors.

    Statistics are pooled over samples and all non-channel axes of the
    training split and accumulated in a streaming fashion (float64), so large
    image sets need not fit in memory. Modes: '0mean' (z-score), '01'
    (min-max) and 'none'. ``state_dict()`` / ``load_state_dict()`` make it
    checkpoint-safe (plain lists, JSON-serializable).

    ``transform`` / ``inverse_transform`` act on one sample ``(C, ...)``;
    ``transform_batch`` / ``inverse_transform_batch`` act on ``(B, C, ...)``.
    """

    def __init__(self, mode: str = "0mean", kind: str = "signal1d"):
        mode = str(mode).lower()
        if mode not in ("0mean", "01", "none"):
            raise ValueError(f"ChannelNormalizer mode must be '0mean', '01' or 'none'; got {mode!r}.")
        self.mode = mode
        self.kind = str(kind)
        self._shift: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, tensors: Iterable[torch.Tensor], channel_dim: int = 0) -> "ChannelNormalizer":
        """Fit on samples ``(C, ...)`` (``channel_dim=0``) or batches ``(B, C, ...)`` (``channel_dim=1``)."""
        if self.mode != "none":
            count = 0
            total = total_sq = vmin = vmax = None
            for tensor in tensors:
                t = torch.as_tensor(tensor).detach().cpu().double()
                flat = t.movedim(channel_dim, 0).reshape(t.shape[channel_dim], -1)
                if total is None:
                    total = torch.zeros(flat.shape[0], dtype=torch.float64)
                    total_sq = torch.zeros_like(total)
                    vmin = torch.full_like(total, float("inf"))
                    vmax = torch.full_like(total, float("-inf"))
                elif flat.shape[0] != total.shape[0]:
                    raise ValueError(
                        f"ChannelNormalizer.fit: channel count changed from "
                        f"{total.shape[0]} to {flat.shape[0]}."
                    )
                count += flat.shape[1]
                total += flat.sum(dim=1)
                total_sq += flat.square().sum(dim=1)
                vmin = torch.minimum(vmin, flat.min(dim=1).values)
                vmax = torch.maximum(vmax, flat.max(dim=1).values)
            if total is None:
                raise ValueError("ChannelNormalizer.fit needs at least one tensor.")
            if self.mode == "0mean":
                shift = total / count
                scale = (total_sq / count - shift.square()).clamp_min(0.0).sqrt()
            else:
                shift = vmin
                scale = vmax - vmin
            self._shift = shift.numpy().astype(np.float32)
            self._scale = np.where(scale.numpy() > 1e-8, scale.numpy(), 1.0).astype(np.float32)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def _params(self, tensor: torch.Tensor, channel_dim: int):
        shape = [1] * tensor.ndim
        shape[channel_dim] = -1
        shift = torch.from_numpy(self._shift).to(device=tensor.device).view(shape)
        scale = torch.from_numpy(self._scale).to(device=tensor.device).view(shape)
        return shift, scale

    def _apply(self, tensor: torch.Tensor, channel_dim: int, inverse: bool) -> torch.Tensor:
        if self.mode == "none" or self._shift is None:
            return tensor.float()
        shift, scale = self._params(tensor, channel_dim)
        x = tensor.float()
        return (x * scale + shift) if inverse else ((x - shift) / scale)

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._apply(tensor, 0, inverse=False)

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._apply(tensor.detach().cpu(), 0, inverse=True)

    def transform_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._apply(tensor, 1, inverse=False)

    def inverse_transform_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._apply(tensor, 1, inverse=True)

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "kind": self.kind,
            "mode": self.mode,
            "shift": self._shift.tolist() if self._shift is not None else None,
            "scale": self._scale.tolist() if self._scale is not None else None,
            "fitted": self._fitted,
        }

    def load_state_dict(self, d: dict) -> None:
        self.mode = d["mode"]
        self.kind = str(d.get("kind", self.kind))
        self._shift = np.array(d["shift"], dtype=np.float32) if d.get("shift") is not None else None
        self._scale = np.array(d["scale"], dtype=np.float32) if d.get("scale") is not None else None
        self._fitted = bool(d.get("fitted", False))

    def __repr__(self) -> str:
        channels = len(self._shift) if self._shift is not None else "N/A"
        return f"ChannelNormalizer(mode={self.mode!r}, kind={self.kind!r}, C={channels}, fitted={self._fitted})"

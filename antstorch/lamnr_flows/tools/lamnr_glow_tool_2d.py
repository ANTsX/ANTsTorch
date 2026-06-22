#!/usr/bin/env python3
"""
lamnr_glow_tool_2d.py — LAM-Flow (Glow 2D) Inference & Analysis Toolkit

Thin shim over lamnr_glow_tool_base.GlowToolBase.
All shared logic (gauss-fit, gauss-impute, recon-template, recon-interpolate,
calc-distance, sample …) lives in the base. This file implements only the
2D-specific I/O hooks:
  - NIfTI slice extraction via ANTs or PIL for 2D images
  - 4D (B, C, H, W) tensor coercion
  - build_model_from_config_2d
  - _prime_if_needed_2d
  - edit_latents_to_mean (2D variant)
  - export-slices (2D-only subcommand)

v0.3.9-refactored
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ants
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from PIL import Image

try:
    from antstorch import create_glow_normalizing_flow_model_2d
except ImportError:
    print("[warn] 'antstorch' not found. Ensure it is installed for 2D Glow models.")
    create_glow_normalizing_flow_model_2d = None

# Import everything from the shared base
from antstorch.lamnr_flows.core.lamnr_glow_tool_base import GlowToolBase

__version__ = "0.3.9"

# ─────────────────────────────────────────────────────────────────────────────
# 2D-only parsers
# ─────────────────────────────────────────────────────────────────────────────

def parse_hw(spec: str) -> Tuple[int, int]:
    try:
        a, b = spec.lower().split("x")
        H, W = int(a), int(b)
        assert H > 0 and W > 0
        return H, W
    except Exception:
        raise argparse.ArgumentTypeError(
            f"Invalid HxW spec '{spec}'. Expected like '128x128'."
        )


def parse_hw_float(spec: str) -> Tuple[float, float]:
    try:
        a, b = spec.lower().split("x")
        H, W = float(a), float(b)
        assert H > 0 and W > 0
        return H, W
    except Exception:
        raise argparse.ArgumentTypeError(
            f"Invalid spacing spec '{spec}'. Expected like '0.8x0.8'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2D-specific I/O helpers (module-level, unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def _coerce_nchw_4d(x, target_hw=None):
    """Coerce any tensor/list output to (B, C, H, W) float32."""
    if isinstance(x, (list, tuple)):
        cands = [t for t in x if torch.is_tensor(t) and t.dim() in (3, 4)]
        if not cands:
            raise RuntimeError("Sample output is not a tensor.")
        areas, fixed = [], []
        for t in cands:
            if t.dim() == 3:
                if t.shape[-1] in (1, 3) and t.shape[0] not in (1, 3):
                    t = t.permute(2, 0, 1).contiguous()
                t = t.unsqueeze(0)
            elif t.dim() == 4 and t.shape[-1] in (1, 3) and t.shape[1] not in (1, 3):
                t = t.permute(0, 3, 1, 2).contiguous()
            fixed.append(t)
            areas.append(int(t.shape[-1]) * int(t.shape[-2]))
        x = fixed[int(torch.tensor(areas).argmax().item())]

    if not torch.is_tensor(x):
        raise RuntimeError(f"Unexpected sample output type: {type(x)}")

    if x.dim() == 3:
        if x.shape[-1] in (1, 3) and x.shape[0] not in (1, 3):
            x = x.permute(2, 0, 1).contiguous()
        x = x.unsqueeze(0)
    if x.dim() == 4 and x.shape[-1] in (1, 3) and x.shape[1] not in (1, 3):
        x = x.permute(0, 3, 1, 2).contiguous()

    if x.size(1) not in (1, 3):
        x = x.mean(dim=1, keepdim=True)
    x = x.float()
    try:
        if x.amin() < 0.0 or x.amax() > 1.0:
            x = to01(x)
    except Exception:
        pass

    if target_hw is not None:
        Ht, Wt = int(target_hw[0]), int(target_hw[1])
        if (x.shape[-2], x.shape[-1]) != (Ht, Wt):
            x = F.interpolate(x, size=(Ht, Wt), mode="bilinear", align_corners=False)
    return x


def _read_image_any(
    path: Path,
    slice_axis: int,
    slice_index: int,
    target_hw: Optional[Tuple[int, int]] = None,
    mask_background: bool = False,
) -> torch.Tensor:
    """
    Read a 2D image from disk → (1, H, W) or (3, H, W) float32.
    Handles NIfTI (slice extraction) and PNG/JPG/TIFF.
    """
    path = Path(path)
    ext  = path.suffix.lower()
    is_nii = ext == ".nii" or (ext == ".gz" and path.name.endswith(".nii.gz"))

    if is_nii:
        img = ants.image_read(str(path))
        try:
            shp = tuple(int(v) for v in img.shape)
        except Exception:
            shp = None

        ax  = int(slice_axis)
        idx = int(slice_index)
        if shp is not None and 0 <= ax < len(shp):
            if idx < 0: idx = 0
            if idx >= shp[ax]: idx = shp[ax] // 2
        try:
            img2d = ants.slice_image(img, axis=ax, idx=idx, collapse_strategy=0)
        except Exception:
            mid0 = shp[0] // 2 if (shp is not None and len(shp) > 0) else 0
            img2d = ants.slice_image(img, axis=0, idx=mid0, collapse_strategy=0)

        if mask_background:
            img2d = img2d * ants.threshold_image(
                ants.otsu_segmentation(img2d, 3), 0, 0, 0, 1
            )

        if target_hw is not None:
            H, W = target_hw
            resize_factor = min(
                float(H) / float(img2d.shape[0]),
                float(W) / float(img2d.shape[1]),
            )
            spacing = (
                img2d.spacing[0] / resize_factor,
                img2d.spacing[1] / resize_factor,
            )
            img2d = ants.resample_image(img2d, spacing, use_voxels=False, interp_type=0)
            img2d = ants.pad_or_crop_image_to_size(img2d, (H, W))

        arr = img2d.numpy()
        if img2d.components > 1 or (arr.ndim == 3 and arr.shape[-1] in [3, 4]):
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            t = torch.from_numpy(arr).permute(2, 0, 1).float()  # (3, H, W)
        else:
            if arr.ndim > 2:
                arr = np.squeeze(arr)
            t = torch.from_numpy(arr).float().unsqueeze(0)  # (1, H, W)

        # Min-Max normalise
        mn, mx = float(t.min()), float(t.max())
        if mx > mn:
            t = (t - mn) / (mx - mn + 1e-8)
        return t

    else:
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        t   = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0  # (3, H, W)
        if target_hw is not None:
            t = F.interpolate(t.unsqueeze(0), size=target_hw,
                              mode="bilinear", align_corners=False).squeeze(0)
        return t


def save_grid_2d(
    x: torch.Tensor,
    out_path,
    nrow: int,
    target_hw: Optional[Tuple[int, int]] = None,
    winsorize: bool = True,
):
    """Save a 4D tensor (B, C, H, W) as a PNG grid or NIfTI."""
    x = _coerce_nchw_4d(x, target_hw=target_hw)
    x = to01(x, winsorize=winsorize)
    out_path = Path(out_path)
    ext = "".join(out_path.suffixes).lower()
    if ".nii" in ext:
        arr = x.detach().cpu().numpy()
        if arr.shape[0] > 1:
            arr_ants = np.transpose(arr.squeeze(1), (1, 2, 0))
        else:
            arr_ants = arr[0, 0]
        ants_img = ants.from_numpy(arr_ants)
        ants.image_write(ants_img, str(out_path))
    else:
        tv.utils.save_image(x, str(out_path), nrow=int(nrow))


@torch.no_grad()
def _prime_if_needed_2d(model, H: int, W: int, device: torch.device):
    """One dummy pass to initialise ActNorm statistics."""
    x = torch.zeros(1, 1, int(H), int(W), device=device, dtype=torch.float32)
    try:
        model.inverse_and_log_det(x)
    except Exception:
        try:
            model.log_prob(x)
        except Exception:
            pass


def build_model_from_config_2d(cfg: dict, device: torch.device, target_hw=None):
    """Build 2D Glow model from checkpoint config dict."""
    if target_hw is not None:
        H, W = int(target_hw[0]), int(target_hw[1])
    else:
        H = int(cfg.get("H", 128))
        W = int(cfg.get("W", 128))
    C = int(cfg.get("C", 1))  # flow models usually have C=1
    input_shape = (C, H, W)

    if create_glow_normalizing_flow_model_2d is None:
        raise ImportError("antstorch.create_glow_normalizing_flow_model_2d is required.")

    raw_k = cfg.get("K", 32)
    parsed_k = [int(v) for v in raw_k] if isinstance(raw_k, (list, tuple)) else int(raw_k)
    raw_hidden = cfg.get("hidden", 128)
    parsed_hidden = ([int(v) for v in raw_hidden]
                     if isinstance(raw_hidden, (list, tuple)) else int(raw_hidden))

    m = create_glow_normalizing_flow_model_2d(
        input_shape=input_shape,
        L=int(cfg.get("L", 4)),
        K=parsed_k,
        hidden_channels=parsed_hidden,
        base=str(cfg.get("base", "glow")),
        glowbase_logscale_factor=float(cfg.get("glowbase_logscale_factor", 3.0)),
        glowbase_min_log=float(cfg.get("glowbase_min_log", -5.0)),
        glowbase_max_log=float(cfg.get("glowbase_max_log", 5.0)),
        split_mode="channel",
        scale=True,
        scale_map=str(cfg.get("scale_map", "tanh")),
        leaky=0.0,
        net_actnorm=bool(cfg.get("net_actnorm", False)),
        scale_cap=float(cfg.get("scale_cap", 2.0)),
    ).to(device).float().eval()

    if not hasattr(m, "input_shape"):
        m.input_shape = input_shape
    return m


def _edit_latents_to_mean_for_view_2d(
    z_list: List[torch.Tensor],
    gauss_blob: Dict,
    view_name: str,
    levels_to_edit: List[int],
    mode: str = "mean",
    pc_index: int = 0,
    pc_scale: float = 2.0,
    pc_center: str = "sample",
    pc_k: int = 64,
    pc_beta: float = 0.0,
) -> List[torch.Tensor]:
    """
    Edit latents for 2D model (4D tensors: B, C, H, W).
    Mirror of the 3D version but asserts ndim == 4.
    """
    if not levels_to_edit:
        return z_list

    views, dims_tbl, shapes_by_view, L = _validate_gauss_blob(gauss_blob)
    try:
        v_idx = views.index(view_name)
    except ValueError:
        raise RuntimeError(f"[recon] View '{view_name}' not in Gaussian header {views}.")

    mu_list    = gauss_blob["mu"]
    Sigma_list = gauss_blob.get("Sigma", None)

    raw_slices = gauss_blob.get("level_view_slices", None)
    V = len(views)
    level_view_slices = []
    if raw_slices is not None:
        for l in range(L):
            row = raw_slices[l]
            if isinstance(row, dict):
                level_view_slices.append({int(k): tuple(v) for k, v in row.items()})
            else:
                level_view_slices.append({vi: tuple(row[vi]) for vi in range(V)})
    else:
        for l in range(L):
            off, row_int = 0, {}
            for vi in range(V):
                d = int(np.asarray(dims_tbl[vi][l]).item()
                        if hasattr(dims_tbl[vi][l], "item") else dims_tbl[vi][l])
                row_int[vi] = (off, off + d)
                off += d
            level_view_slices.append(row_int)

    levels_set = {int(l) for l in levels_to_edit}
    z_out: List[torch.Tensor] = []

    for l, z_l in enumerate(z_list):
        if l not in levels_set:
            z_out.append(z_l)
            continue

        if z_l.ndim != 4:
            raise RuntimeError(f"[recon] Expected 4D latent at level {l}, got {z_l.shape}.")

        B, C, H, W = z_l.shape
        Cg, Hg, Wg = shapes_by_view[v_idx][l]
        if (C, H, W) != (Cg, Hg, Wg):
            raise RuntimeError(
                f"[recon] Shape mismatch level {l}, view '{view_name}': "
                f"model ({C},{H},{W}) vs Gauss ({Cg},{Hg},{Wg})."
            )

        a, b = level_view_slices[l][v_idx]
        mu_level   = np.asarray(mu_list[l], dtype=np.float64).ravel()
        mu_view_flat = mu_level[a:b]
        mu_view = torch.as_tensor(
            mu_view_flat, dtype=z_l.dtype, device=z_l.device
        ).view(1, C, H, W)

        if mode == "mean":
            z_l_edit = mu_view.expand(B, C, H, W)

        elif mode == "zero":
            z_l_edit = torch.zeros_like(z_l)

        elif mode in ("pc", "pc_denoise"):
            Sigma_l = (Sigma_list[l] if isinstance(Sigma_list, (list, tuple))
                       else Sigma_list)
            Dv = C * H * W
            if isinstance(Sigma_l, dict) and Sigma_l.get("type") == "lowrank":
                U      = np.asarray(Sigma_l["U"], dtype=np.float64)
                eig    = np.asarray(Sigma_l["eig"], dtype=np.float64)
                sigma2 = float(Sigma_l.get("sigma2", 0.0))
                U_v    = U[a:b, :]
                Sv     = (U_v * eig[np.newaxis, :]) @ U_v.T
                if sigma2 > 0.0:
                    Sv += sigma2 * np.eye(Dv, dtype=np.float64)
            else:
                S = np.asarray(Sigma_l, dtype=np.float64)
                Sv = np.diag(S[a:b]) if S.ndim == 1 else S[a:b, a:b]

            Sv = 0.5 * (Sv + Sv.T)
            w_eig, V_mat = np.linalg.eigh(Sv)

            if mode == "pc":
                k   = int(pc_index)
                col = -1 - k
                direction_np = V_mat[:, col]
                lam  = float(max(w_eig[col], 0.0))
                step = float(pc_scale) * (lam ** 0.5 if lam > 0.0 else 0.0)
                direction_t = torch.from_numpy(
                    direction_np.astype(np.float32)
                ).view(1, C, H, W).to(z_l.device, z_l.dtype)
                base = (mu_view.expand(B, C, H, W)
                        if pc_center.lower() == "mean" else z_l)
                z_l_edit = base + step * direction_t
                print(f"[recon] level {l}, '{view_name}': PC{pc_index} "
                      f"λ={lam:.3e}, step={step:.3e}, center={pc_center}")
            else:  # pc_denoise
                V_desc  = V_mat[:, ::-1]
                k_keep  = min(max(int(pc_k), 0), V_desc.shape[1])
                V_t     = torch.from_numpy(V_desc.astype(np.float32)).to(z_l.device, z_l.dtype)
                z_flat  = z_l.view(B, -1)
                mu_flat = mu_view.view(1, -1)
                y = torch.matmul(z_flat - mu_flat, V_t)
                if k_keep < V_t.shape[1]:
                    tail = y[:, k_keep:]
                    y[:, k_keep:] = 0.0 if float(pc_beta) == 0.0 else float(pc_beta) * tail
                z_flat_edit = mu_flat + torch.matmul(y, V_t.T)
                z_l_edit    = z_flat_edit.view(B, C, H, W)
                print(f"[recon] level {l}, '{view_name}': "
                      f"pc_denoise k_keep={k_keep}, β={pc_beta:.3f}")
        else:
            raise ValueError(f"[recon] Unknown edit mode '{mode}'.")

        z_out.append(z_l_edit)
    return z_out


# ANTs-based 2D resampling helpers (preserved verbatim)

@torch.no_grad()
def resample_with_ants_spacing(
    x: torch.Tensor,
    native_spacing: Tuple[float, float],
    target_spacing: Tuple[float, float],
) -> torch.Tensor:
    device, dtype = x.device, x.dtype
    N, C = x.shape[0], x.shape[1]
    outs = []
    for c in range(C):
        xs = []
        for i in range(N):
            arr = x[i, c].detach().cpu().numpy()
            img = ants.from_numpy(arr)
            try:
                img.set_spacing((float(native_spacing[0]), float(native_spacing[1])))
            except Exception:
                img.spacing = (float(native_spacing[0]), float(native_spacing[1]))
            img_r = ants.resample_image(
                img,
                (float(target_spacing[0]), float(target_spacing[1])),
                use_voxels=False,
                interp_type=0,
            )
            xs.append(torch.from_numpy(img_r.numpy()).to(device=device, dtype=dtype))
        outs.append(torch.stack(xs, dim=0))
    return torch.stack(outs, dim=1)


@torch.no_grad()
def resample_with_ants_size(
    x: torch.Tensor,
    target_size: Tuple[int, int],
    native_spacing: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    device, dtype = x.device, x.dtype
    N, C = x.shape[0], x.shape[1]
    outs = []
    for c in range(C):
        xs = []
        for i in range(N):
            arr = x[i, c].detach().cpu().numpy()
            img = ants.from_numpy(arr)
            if native_spacing is not None:
                try:
                    img.set_spacing((float(native_spacing[0]), float(native_spacing[1])))
                except Exception:
                    img.spacing = (float(native_spacing[0]), float(native_spacing[1]))
            img_r = ants.resample_image(
                img,
                (int(target_size[0]), int(target_size[1])),
                use_voxels=True,
                interp_type=0,
            )
            xs.append(torch.from_numpy(img_r.numpy()).to(device=device, dtype=dtype))
        outs.append(torch.stack(xs, dim=0))
    return torch.stack(outs, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# GlowTool2D — concrete subclass
# ─────────────────────────────────────────────────────────────────────────────

class GlowTool2D(GlowToolBase):
    """
    2D Glow inference tool.

    Spatial tensors: (B, C, H, W)  — ndim == 4.
    Size spec:       "HxW" parsed via parse_hw().
    CLI size args:   --slice-axis INT --slice-index INT
                     (image is extracted from a 3D NIfTI volume at runtime)
    """

    # ── Abstract properties ──────────────────────────────────────────────── #

    @property
    def ndim(self) -> int:
        return 4

    @property
    def interp_mode(self) -> str:
        return "bilinear"

    @property
    def default_cov_estimator(self) -> str:
        return "full"

    @property
    def default_cov_rank(self) -> int:
        return 64

    # ── Abstract I/O methods ─────────────────────────────────────────────── #

    def parse_size(self, spec: str) -> Tuple[int, int]:
        return parse_hw(spec)

    def parse_spacing(self, spec: str) -> Tuple[float, float]:
        return parse_hw_float(spec)

    def read_image(self, path: "Path", target_size, **kw) -> torch.Tensor:
        """Read 2D image from NIfTI/PNG at the stored slice position."""
        slice_axis  = kw.get("slice_axis",  self._default_slice_axis)
        slice_index = kw.get("slice_index", self._default_slice_index)
        t = _read_image_any(
            path, slice_axis, slice_index,
            target_hw=target_size if target_size else None,
            mask_background=kw.get("mask_background", False),
        )
        # Normalise channels to 1 (flow model expects C=1)
        if t.shape[0] > 1:
            t = t.mean(dim=0, keepdim=True)
        return t  # (1, H, W)

    def coerce_nd(self, x, target_size) -> torch.Tensor:
        return _coerce_nchw_4d(x, target_hw=target_size)

    def save_volume(self, x: torch.Tensor, path, **kw):
        nrow = kw.get("nrow", 1)
        target_hw = kw.get("target_hw", None)
        save_grid_2d(x, path, nrow=nrow, target_hw=target_hw)

    def build_model(self, cfg: dict, device: torch.device, target_size) -> nn.Module:
        return build_model_from_config_2d(cfg, device, target_hw=target_size)

    def prime_if_needed(self, model, size, device):
        H, W = int(size[0]), int(size[1])
        _prime_if_needed_2d(model, H, W, device)

    def edit_latents_to_mean(
        self,
        z_list: List[torch.Tensor],
        gauss_blob: dict,
        view_name: str,
        levels_to_edit: List[int],
        **kw,
    ) -> List[torch.Tensor]:
        return _edit_latents_to_mean_for_view_2d(
            z_list, gauss_blob, view_name, levels_to_edit, **kw
        )

    # ── 2D-specific CLI hooks ────────────────────────────────────────────── #

    # Default slice parameters (overridden by --slice-axis / --slice-index)
    _default_slice_axis  = 2
    _default_slice_index = 130

    def _add_size_arg(self, ap: argparse.ArgumentParser, required: bool = True):
        """Add 2D-specific slice args to the argparse instance."""
        ap.add_argument(
            "--slice-axis", type=int,
            default=self._default_slice_axis,
            required=False,
            help="Axis along which to slice 3D NIfTI volumes (default 2 = axial).",
        )
        ap.add_argument(
            "--slice-index", type=int,
            default=self._default_slice_index,
            required=False,
            help="Slice index to extract from 3D NIfTI volumes (default 130).",
        )

    def _get_target_size(self, args, cfg: dict) -> Tuple[int, int]:
        """
        For 2D, target spatial size comes from the checkpoint config (H, W).
        The CLI does NOT take --volume-size; spatial size is determined by the model.
        """
        H = int(cfg.get("H", 128))
        W = int(cfg.get("W", 128))
        # Store slice parameters for read_image calls
        self._default_slice_axis  = int(getattr(args, "slice_axis",  self._default_slice_axis))
        self._default_slice_index = int(getattr(args, "slice_index", self._default_slice_index))
        return (H, W)

    def _add_sampling_size_arg(self, ap: argparse.ArgumentParser):
        """2D sample subcommand also needs --image-size for the saved PNG tile size."""
        ap.add_argument("--image-size", type=parse_hw, default="128x128",
                        help="Tile size HxW for saved PNG (default 128x128).")

    # ── 2D-only subcommand: export-slices ───────────────────────────────── #

    def cmd_export_slices(self, argv=None):
        """Export 2D slices from all NIfTI volumes in a manifest."""
        ap = argparse.ArgumentParser("export-slices")
        ap.add_argument("--manifest",      type=str, required=True)
        ap.add_argument("--views",         type=str, required=True)
        ap.add_argument("--outdir",        type=str, required=True)
        ap.add_argument("--output-format", type=str, default="png",
                        choices=["png", "jpg", "nii", "nii.gz"])
        ap.add_argument("--image-size",    type=parse_hw, default="128x128")
        self._add_size_arg(ap, required=False)
        args = ap.parse_args(argv)

        self._default_slice_axis  = int(args.slice_axis)
        self._default_slice_index = int(args.slice_index)

        out_dir = Path(args.outdir)
        out_dir.mkdir(parents=True, exist_ok=True)

        from antstorch.lamnr_flows.core.lamnr_glow_tool_base import _read_manifest_csv
        cols = _read_manifest_csv(Path(args.manifest))
        view_names = [v.strip() for v in args.views.split(",") if v.strip()]

        for vname in view_names:
            if vname not in cols:
                print(f"[error] View '{vname}' not found in manifest.")
                continue
            paths = cols[vname]
            print(f"[export-slices] Exporting {len(paths)} slices for view '{vname}'")
            for i, p in enumerate(paths):
                pth = Path(p)
                if not pth.exists():
                    print(f"[warn] Missing file: {pth}")
                    continue
                xi = _read_image_any(
                    pth, args.slice_axis, args.slice_index, target_hw=args.image_size
                )
                out_name = f"{i:06d}_{vname}.{args.output_format}"
                save_grid_2d(xi.unsqueeze(0), out_dir / out_name, nrow=1,
                             target_hw=args.image_size)

        print(f"[export-slices] Done → {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Backwards-compatibility: expose module-level main_* aliases pointing to the
# class methods so that any code importing old main_* functions still works.
# ─────────────────────────────────────────────────────────────────────────────

_tool = GlowTool2D()

main_gauss_fit            = _tool.cmd_gauss_fit
main_gauss_impute         = _tool.cmd_gauss_impute
main_recon                = _tool.cmd_recon
main_recon_template       = _tool.cmd_recon_template
main_recon_cohort_template= _tool.cmd_recon_cohort_template
main_recon_temperature    = _tool.cmd_recon_temperature
main_recon_interpolate    = _tool.cmd_recon_interpolate
main_calc_distance        = _tool.cmd_calc_distance
main_sample               = _tool.cmd_sample
main_export_slices        = _tool.cmd_export_slices


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _tool.run()

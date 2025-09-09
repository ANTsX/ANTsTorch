"""
Multi-view DataFrame dataset for PyTorch / ANTsTorch.

This module provides a production-ready `MultiViewDataFrameDataset` that wraps one or
more pandas DataFrames ("views") into a single PyTorch Dataset with:

* Per-view normalization (None, z-score "0mean", or min–max "01" with automatic [0,1] clamping)
* Optional Gaussian augmentation in raw or normalized space
* One-hot encoding for categorical columns (numeric-only normalization)
* Finite-value masks and simple imputation (none/mean/zero)
* Deterministic indexing (use DataLoader for shuffling), worker seeding helper
* Optional concatenation of all views into a single feature vector
* Target support aligned to the intersection of indices across views

Intended usage matches ANTsTorch patterns: compute statistics once at init-time,
apply light, explicit transforms in `__getitem__`, and return tensors ready for models.

Example
-------
>>> # Single view (concat to a flat vector)
>>> ds = MultiViewDataFrameDataset(
...     views={"t1": df_t1},
...     target=y,                      # optional, Series or DataFrame
...     normalization="0mean",        # or "01" or None
...     alpha=0.02,
...     add_noise_in="normalized",    # 'raw', 'normalized', or 'none'
...     impute="mean",
...     concat_views=True,
... )
>>> dl = torch.utils.data.DataLoader(
...     ds, batch_size=64, shuffle=True, num_workers=4,
...     worker_init_fn=MultiViewDataFrameDataset.worker_init_fn)
>>> batch = next(iter(dl))
>>> batch['x'].shape, batch['mask'].shape

>>> # Multi-view (keep per-view tensors)
>>> ds = MultiViewDataFrameDataset(
...     views={"t1": df_t1, "dt": df_dt, "rsf": df_rsf},
...     normalization={"t1": "0mean", "dt": "01", "rsf": None},
...     concat_views=False,
... )
>>> batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=8)))
>>> batch['views']['t1'].shape, batch['masks']['t1'].shape

Notes
-----
* Indices: rows are aligned on the intersection of indices across views (and target).
* Numeric vs. categorical: numeric columns are normalized; categorical columns are
  one-hot encoded and passed through unchanged (except for optional imputation).
* "01" normalization includes automatic clamping to [0,1] after optional noise.
* For reproducible augmentation across workers, use `worker_init_fn`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Callable, List, Union, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ------------------------------
# Utilities
# ------------------------------

def _to_tensor(x: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32, copy=False)).to(dtype)


def _finite_mask(a: np.ndarray) -> np.ndarray:
    return np.isfinite(a).astype(np.float32)


@dataclass
class _ViewState:
    columns: List[str]
    numeric_cols: List[str]
    cat_cols: List[str]
    mean: np.ndarray
    std: np.ndarray
    vmin: np.ndarray
    vmax: np.ndarray
    eps_std: np.ndarray
    eps_rng: np.ndarray


class MultiViewDataFrameDataset(Dataset):
    """
    Multi-view (or single-view) tabular dataset for ANTsTorch workflows.

    This dataset handles aligned tabular data across one or more "views" (e.g., modalities or
    feature sets), providing normalization, augmentation, missing-data masking, and optional
    concatenation.

    Parameters
    ----------
    views : Dict[str, pandas.DataFrame]
        Mapping from view name to DataFrame. Rows are samples; indices should be subject IDs.
        For a single view, simply pass {"viewname": dataframe}.

    target : Optional[Union[pandas.Series, pandas.DataFrame]], default=None
        Optional target aligned to the intersection of all view indices. Cast to float.

    normalization : Union[str, Dict[str, Optional[str]]], default='0mean'
        Normalization mode per view: one of {None, '01', '0mean'}. Provide a single string for
        all views or a dict keyed by view name.

    alpha : float, default=0.01
        Gaussian noise scale for augmentation. Noise is applied to **all numeric columns**.

    add_noise_in : {'raw','normalized','none'}, default='raw'
        Where to add Gaussian noise. In 'raw', per-column std is used (`alpha * eps_std`);
        in 'normalized', unit-variance noise with scale `alpha` is used.

    impute : {'none','mean','zero'}, default='none'
        Imputation strategy applied after normalization. For '0mean', 'mean' imputation maps to 0.
        For '01', 'mean' imputation maps to 0 (min in [0,1]).

    concat_views : bool, default=False
        If True, return a single concatenated feature tensor. Otherwise, return per-view tensors.

    column_info : Optional[Dict[str, Dict[str, List[str]]]], default=None
        Optional per-view schema: {'view': {'numeric': [...], 'categorical': [...]}}. If omitted,
        dtypes are auto-detected.

    preprocessors : Optional[Dict[str, Callable[[pandas.DataFrame], pandas.DataFrame]]], default=None
        Optional per-view callable applied before fitting (e.g., custom encoders). Must be deterministic.

    number_of_samples : Optional[int], default=None
        Dataset length; defaults to number of aligned rows (N). If larger, indexing wraps via modulo.

    dtype : torch.dtype, default=torch.float32
        Output tensor dtype.

    Returns (per __getitem__)
    -------------------------
    If concat_views=True:
        {'x': Tensor[D], 'mask': Tensor[D], 'target': Optional[Tensor], 'index': Any}
    else:
        {'views': Dict[str, Tensor[D_v]], 'masks': Dict[str, Tensor[D_v]], 'target': Optional[Tensor], 'index': Any}

    Design Notes (ANTsTorch-style)
    ------------------------------
    * Statistics are computed once at initialization.
    * Augmentation is simple and explicit; heavy transforms belong in model space.
    * Deterministic indexing; use DataLoader's `shuffle`/samplers for randomness.
    * Categorical handling is minimal and predictable (one-hot; no target leakage).
    """

    def __init__(
        self,
        views: Dict[str, pd.DataFrame],
        target: Optional[Union[pd.Series, pd.DataFrame]] = None,
        normalization: Union[str, Dict[str, Optional[str]]] = '0mean',
        alpha: float = 0.01,
        add_noise_in: str = 'raw',
        impute: str = 'none',
        concat_views: bool = False,
        column_info: Optional[Dict[str, Dict[str, List[str]]]] = None,
        preprocessors: Optional[Dict[str, Callable[[pd.DataFrame], pd.DataFrame]]] = None,
        number_of_samples: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
    ):
        assert add_noise_in in {'raw', 'normalized', 'none'}
        assert impute in {'none', 'mean', 'zero'}

        self.view_names = sorted(list(views.keys()))
        self.add_noise_in = add_noise_in
        self.alpha = float(alpha)
        self.impute = impute
        self.concat_views = concat_views
        self.dtype = dtype

        # Normalization config and optional preprocessors
        self._preprocessors = preprocessors or {}
        self._norm_mode: Dict[str, Optional[str]] = self._expand_norm_modes(normalization)

        # Preprocess and collect dataframes
        processed: Dict[str, pd.DataFrame] = {}
        for v in self.view_names:
            df = views[v]
            if v in self._preprocessors:
                df = self._preprocessors[v](df)
            processed[v] = df

        # Align on common index across views (inner join of indices)
        common_index = None
        for v, df in processed.items():
            idx = df.index
            common_index = idx if common_index is None else common_index.intersection(idx)
        if target is not None:
            tgt_idx = target.index if isinstance(target, (pd.Series, pd.DataFrame)) else pd.Index([])
            common_index = common_index.intersection(tgt_idx)
        common_index = common_index.sort_values()
        if len(common_index) == 0:
            raise ValueError("No overlapping indices across views (and target, if provided).")

        # Restrict all to common index and remember order
        self.index = common_index
        aligned: Dict[str, pd.DataFrame] = {v: processed[v].loc[self.index] for v in self.view_names}
        if target is not None:
            if isinstance(target, pd.Series):
                self.target = target.loc[self.index].astype(np.float32).to_numpy()
            elif isinstance(target, pd.DataFrame):
                self.target = target.loc[self.index].astype(np.float32).to_numpy()
            else:
                raise TypeError("target must be a pandas Series or DataFrame if provided.")
        else:
            self.target = None

        # Detect columns and build per-view state
        self._state: Dict[str, _ViewState] = {}
        self._arrs: Dict[str, np.ndarray] = {}

        for v in self.view_names:
            dfv = aligned[v]
            # Column typing
            if column_info and v in column_info:
                numeric_cols = column_info[v].get('numeric', [])
                cat_cols = column_info[v].get('categorical', [])
            else:
                numeric_cols = [c for c in dfv.columns if pd.api.types.is_numeric_dtype(dfv[c])]
                cat_cols = [c for c in dfv.columns if c not in numeric_cols]

            # One-hot encode categoricals (deterministic columns ordering)
            if len(cat_cols) > 0:
                df_cat = pd.get_dummies(dfv[cat_cols].astype('category'), dummy_na=False)
            else:
                df_cat = pd.DataFrame(index=dfv.index)
            df_num = dfv[numeric_cols].astype(np.float32)
            df_enc = pd.concat([df_num, df_cat], axis=1)

            columns = list(df_enc.columns)

            # Compute stats on encoded frame (normalize numeric + leave dummies as-is)
            enc_vals = df_enc.to_numpy(copy=False).astype(np.float32)
            # Identify positions of numeric columns in encoded frame
            num_positions = [columns.index(c) for c in df_num.columns]
            num_vals = enc_vals[:, num_positions] if len(num_positions) > 0 else np.empty((enc_vals.shape[0], 0), dtype=np.float32)

            if num_vals.size > 0:
                mean = np.nanmean(num_vals, axis=0)
                std = np.nanstd(num_vals, axis=0)
                vmin = np.nanmin(num_vals, axis=0)
                vmax = np.nanmax(num_vals, axis=0)
                eps_std = np.maximum(std, 1e-8)
                eps_rng = np.maximum(vmax - vmin, 1e-8)
            else:
                # No numeric cols: create empty arrays
                mean = std = vmin = vmax = eps_std = eps_rng = np.empty((0,), dtype=np.float32)

            self._state[v] = _ViewState(
                columns=columns,
                numeric_cols=list(df_num.columns),
                cat_cols=cat_cols,
                mean=mean.astype(np.float32, copy=False),
                std=std.astype(np.float32, copy=False),
                vmin=vmin.astype(np.float32, copy=False),
                vmax=vmax.astype(np.float32, copy=False),
                eps_std=eps_std.astype(np.float32, copy=False),
                eps_rng=eps_rng.astype(np.float32, copy=False),
            )

            # Persist encoded array for fast row access
            self._arrs[v] = enc_vals

        self.N = len(self.index)
        self.number_of_samples = int(number_of_samples) if number_of_samples is not None else self.N

        # Pre-compute per-view mapping from numeric positions for speed
        self._num_pos: Dict[str, np.ndarray] = {}
        self._dummy_pos: Dict[str, np.ndarray] = {}
        for v in self.view_names:
            st = self._state[v]
            if len(st.numeric_cols) > 0:
                num_pos = np.array([self._state[v].columns.index(c) for c in st.numeric_cols], dtype=np.int64)
            else:
                num_pos = np.array([], dtype=np.int64)
            dummy_pos = np.array([i for i, c in enumerate(st.columns) if c not in st.numeric_cols], dtype=np.int64)
            self._num_pos[v] = num_pos
            self._dummy_pos[v] = dummy_pos

        # jitter positions removed (noise applied to all numeric columns)

    def __len__(self) -> int:
        return self.number_of_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return one sample by aligned index (modulo N).

        Steps for numeric features:
          raw-noise (optional) -> normalization -> normalized-noise (optional)
          -> auto clamp to [0,1] if mode == '01' -> imputation (optional).
        """
        if torch.is_tensor(idx):
            idx = int(idx.item())
        ridx = idx % self.N
        index_value = self.index[ridx]

        view_tensors: Dict[str, torch.Tensor] = {}
        view_masks: Dict[str, torch.Tensor] = {}
        concat_list: List[np.ndarray] = []
        concat_mask_list: List[np.ndarray] = []

        for v in self.view_names:
            x = self._arrs[v][ridx].copy()  # (D_v,)
            st = self._state[v]
            num_pos = self._num_pos[v]
            dummy_pos = self._dummy_pos[v]

                        # --- Normalization and augmentation ---
            if num_pos.size > 0:
                x_num = x[num_pos]

                if self.add_noise_in == 'raw' and self.alpha > 0:
                    x_num = x_num + np.random.normal(0.0, self.alpha * st.eps_std).astype(np.float32)

                
                norm_mode = self._norm_mode[v]
                if norm_mode == '0mean':
                    x_num = (x_num - st.mean) / st.eps_std
                elif norm_mode == '01':
                    x_num = (x_num - st.vmin) / st.eps_rng
                # else: no normalization

                if self.add_noise_in == 'normalized' and self.alpha > 0:
                    x_num = x_num + np.random.normal(0.0, self.alpha, size=x_num.shape).astype(np.float32)

                # Automatic clamp for '01' mode
                if norm_mode == '01':
                    x_num = np.clip(x_num, 0.0, 1.0)

                # Imputation (after normalization)
                if self.impute == 'mean':
                    if norm_mode == '0mean':
                        imp = np.zeros_like(st.mean, dtype=np.float32)  # mean->0 after z-score
                    elif norm_mode == '01':
                        imp = np.zeros_like(st.vmin, dtype=np.float32)  # default to 0 (min) in [0,1]
                    else:
                        imp = st.mean  # raw space mean
                    x_num = np.where(np.isfinite(x_num), x_num, imp)
                elif self.impute == 'zero':
                    x_num = np.where(np.isfinite(x_num), x_num, 0.0)

                x[num_pos] = x_num

            # Dummies are already 0/1; keep as-is. If NaNs appear, handle per impute
            if dummy_pos.size > 0 and self.impute != 'none':
                xd = x[dummy_pos]
                if self.impute == 'mean':
                    xd = np.where(np.isfinite(xd), xd, 0.0)  # mean of one-hot ~ 0 for rare categories
                else:
                    xd = np.where(np.isfinite(xd), xd, 0.0)
                x[dummy_pos] = xd

            # Mask: 1 if finite, else 0
            mask = _finite_mask(x)

            if self.concat_views:
                concat_list.append(x)
                concat_mask_list.append(mask)
            else:
                view_tensors[v] = _to_tensor(x, self.dtype)
                view_masks[v] = _to_tensor(mask, self.dtype)

        if self.concat_views:
            x_all = np.concatenate(concat_list, axis=0)
            m_all = np.concatenate(concat_mask_list, axis=0)
            sample = {
                'x': _to_tensor(x_all, self.dtype),
                'mask': _to_tensor(m_all, self.dtype),
                'target': None if self.target is None else _to_tensor(self.target[ridx].reshape(-1), self.dtype),
                'index': index_value,
            }
        else:
            sample = {
                'views': view_tensors,
                'masks': view_masks,
                'target': None if self.target is None else _to_tensor(self.target[ridx].reshape(-1), self.dtype),
                'index': index_value,
            }
        return sample

    def denormalize(self, view: str, x: np.ndarray) -> np.ndarray:
        """Invert normalization for a single view's **numeric** part.

        Categorical one-hots are returned unchanged. Accepts 1D (D_v,) or 2D (B, D_v).
        """
        st = self._state[view]
        norm_mode = self._norm_mode[view]
        num_pos = self._num_pos[view]
        if norm_mode is None or num_pos.size == 0:
            return x

        out = np.array(x, copy=True)
        if out.ndim == 1:
            x_num = out[num_pos]
            if norm_mode == '0mean':
                x_num = x_num * st.eps_std + st.mean
            elif norm_mode == '01':
                x_num = x_num * st.eps_rng + st.vmin
            out[num_pos] = x_num
        else:
            x_num = out[:, num_pos]
            if norm_mode == '0mean':
                x_num = x_num * st.eps_std[None, :] + st.mean[None, :]
            elif norm_mode == '01':
                x_num = x_num * st.eps_rng[None, :] + st.vmin[None, :]
            out[:, num_pos] = x_num
        return out

    def view_dim(self, view: str) -> int:
        return len(self._state[view].columns)

    def total_dim(self) -> int:
        return int(sum(self.view_dim(v) for v in self.view_names))

    def norm_mode(self, view: str) -> Optional[str]:
        return self._norm_mode[view]

    @staticmethod
    def worker_init_fn(worker_id: int):
        """Seed NumPy for reproducible augmentation across DataLoader workers."""
        base_seed = torch.initial_seed() % 2**32
        np.random.seed(base_seed + worker_id)

    # ------------------------------
    # Internal
    # ------------------------------
    def _expand_norm_modes(self, normalization: Union[str, Dict[str, Optional[str]]]) -> Dict[str, Optional[str]]:
        if isinstance(normalization, str) or normalization is None:
            modes = {v: normalization for v in self.view_names}
        elif isinstance(normalization, dict):
            modes = {v: normalization.get(v, '0mean') for v in self.view_names}
        else:
            raise TypeError("normalization must be str, dict, or None")
        for v, m in modes.items():
            if m not in {None, '01', '0mean'}:
                raise ValueError(f"Unsupported normalization mode '{m}' for view '{v}'")
        return modes



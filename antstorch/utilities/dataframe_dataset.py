# dataframe_dataset_patched.py
# Multi-view DataFrame dataset for PyTorch / ANTsTorch.
# Owns per-view normalization and optional Gaussian jitter (raw or normalized space).
# Adds: set_alpha(), denormalize(), worker_init_fn(), view_dim(), total_dim().
# Designed to be drop-in compatible with previous versions you used.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Callable, List, Union, Any, Tuple

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
    Multi-view (or single-view) tabular dataset for PyTorch.

    This dataset aligns one or more pandas DataFrames by row index and returns
    tensors suitable for training normalizing-flow and other tabular models.
    It optionally normalizes numeric features, injects Gaussian jitter for
    augmentation, one-hot encodes categoricals, and provides finite-value masks.

    Parameters
    ----------
    views : Dict[str, pandas.DataFrame]
        Mapping from view name to DataFrame (rows = samples). All views are aligned
        on the intersection of their indices. For a single view, pass
        ``{"view": df}``.

    target : Optional[Union[pandas.Series, pandas.DataFrame]], default=None
        Optional target aligned to the intersection of indices across all views.
        Converted to float and returned with each sample when provided.

    normalization : Union[str, Dict[str, Optional[str]]], default='0mean'
        Per-view normalization mode: one of ``None``, ``'0mean'`` (z-score),
        or ``'01'`` (min–max to [0,1]). Provide a single string for all views
        or a dict keyed by view name.

    alpha : float, default=0.0
        Scale of Gaussian jitter applied to numeric features. Exact behavior
        depends on ``add_noise_in``.

    add_noise_in : {'raw', 'normalized', 'none'}, default='raw'
        Domain for jitter:
          * ``'raw'`` — noise ~ N(0, (alpha * per-column std)^2) before normalization
          * ``'normalized'`` — noise ~ N(0, alpha^2) after normalization
          * ``'none'`` — no noise

    impute : {'none', 'mean', 'zero'}, default='none'
        Imputation applied after normalization:
          * ``'mean'`` — maps to 0 for ``'0mean'``/``'01'``, raw mean otherwise
          * ``'zero'`` — fills NaNs/inf with 0

    concat_views : bool, default=False
        If ``True``, returns a single concatenated feature vector per sample.
        If ``False``, returns a dict of per-view tensors.

    column_info : Optional[Dict[str, Dict[str, List[str]]]], default=None
        Optional schema per view: ``{'view': {'numeric': [...], 'categorical': [...]}}``.
        If omitted, types are inferred (numeric dtypes vs others).

    preprocessors : Optional[Dict[str, Callable[[pandas.DataFrame], pandas.DataFrame]]], default=None
        Optional deterministic per-view preprocessing callables run before
        statistics are computed.

    number_of_samples : Optional[int], default=None
        Dataset length; defaults to number of aligned rows. If larger than the
        number of rows, indexing wraps (modulo).

    dtype : torch.dtype, default=torch.float32
        Output tensor dtype.

    nonfinite_clean : Optional[str], default=None
        Optional pre-cleaning before stats: one of ``None``, ``'drop'``,
        ``'impute_zero'``, ``'impute_mean'``. Applied consistently across views.

    nonfinite_scope : {'numeric_only', 'all_columns'}, default='numeric_only'
        Columns considered when computing row-level non-finite handling for
        ``nonfinite_clean``.

    max_row_na_frac : float, default=0.0
        For ``nonfinite_clean='drop'``: maximum allowed fraction of non-finite
        values per row within the chosen scope.

    Attributes
    ----------
    view_names : List[str]
        View names in the order provided.

    index : pandas.Index
        Aligned index (intersection) used by all views (and target if provided).

    N : int
        Number of aligned rows.

    number_of_samples : int
        Reported dataset length (may exceed N if wrapping).

    dtype : torch.dtype
        Output dtype.

    alpha : float
        Current jitter scale.

    add_noise_in : str
        Current jitter domain ('raw' | 'normalized' | 'none').

    impute : str
        Current imputation mode.

    concat_views : bool
        Whether samples are concatenated or per-view.

    Methods
    -------
    set_alpha(alpha: float, add_noise_in: Optional[str] = None) -> None
        Adjust jitter magnitude and/or domain at runtime.

    denormalize(view: str, x: np.ndarray) -> np.ndarray
        Invert normalization for the numeric portion of a view (categoricals are
        returned unchanged).

    view_dim(view: str) -> int
        Total encoded dimensionality for a view (numeric + one-hots).

    total_dim() -> int
        Sum of encoded dimensionalities across all views.

    norm_mode(view: str) -> Optional[str]
        Normalization mode in effect for a given view.
        
    worker_init_fn(worker_id: int) -> None
        Seed NumPy in DataLoader workers for reproducible augmentation.

    __getitem__ Returns
    -------------------
    If ``concat_views=True``:
        ``{'x': Tensor[D], 'mask': Tensor[D], 'index': Any, 'target': Optional[Tensor]}``
    If ``concat_views=False``:
        ``{'views': Dict[str, Tensor[D_v]], 'masks': Dict[str, Tensor[D_v]], 'index': Any, 'target': Optional[Tensor]}``

    Notes
    -----
    * Numeric features are normalized; categoricals are one-hot encoded and passed through.
    * For ``'01'`` normalization, outputs are clamped to [0,1] after optional jitter.
    * Statistics are computed once at init on the provided data (per-split if you
      construct separate train/val datasets).

    Examples
    --------
    >>> ds = MultiViewDataFrameDataset(
    ...     views={'t1': df_t1, 'rsf': df_rsf},
    ...     normalization={'t1': '0mean', 'rsf': '01'},
    ...     alpha=0.02, add_noise_in='normalized', impute='mean')
    >>> dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True,
    ...     worker_init_fn=MultiViewDataFrameDataset.worker_init_fn)
    >>> batch = next(iter(dl))
    >>> batch['views']['t1'].shape, batch['masks']['t1'].shape
    """

    def __init__(
        self,
        views: Dict[str, pd.DataFrame],
        target: Optional[Union[pd.Series, pd.DataFrame]] = None,
        normalization: Union[str, Dict[str, Optional[str]]] = '0mean',
        alpha: float = 0.0,
        add_noise_in: str = 'raw',              # 'raw' | 'normalized' | 'none'
        impute: str = 'none',                   # 'none' | 'mean' | 'zero'
        concat_views: bool = False,
        column_info: Optional[Dict[str, Dict[str, List[str]]]] = None,
        preprocessors: Optional[Dict[str, Callable[[pd.DataFrame], pd.DataFrame]]] = None,
        number_of_samples: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        # Non-finite handling before stats (optional, conservative defaults)
        nonfinite_clean: Optional[str] = None,        # None | 'drop' | 'impute_zero' | 'impute_mean'
        nonfinite_scope: str = 'numeric_only',        # 'numeric_only' | 'all_columns'
        max_row_na_frac: float = 0.0,                 # for 'drop': allow up to this frac non-finites per row
    ):
        assert add_noise_in in {'raw', 'normalized', 'none'}
        assert impute in {'none', 'mean', 'zero'}

        self.view_names = list(views.keys())  # preserve caller order
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

        # Align on common index across views (inner join of indices); include target if provided
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
        self.nonfinite_clean = nonfinite_clean
        self.nonfinite_scope = nonfinite_scope
        self.max_row_na_frac = float(max_row_na_frac)

        # Optional pre-clean (drop/impute) for stability
        if nonfinite_clean in {'drop', 'impute_zero', 'impute_mean'}:
            keep_rows = np.ones((len(self.index),), dtype=bool)
            if nonfinite_clean == 'drop':
                # mark rows to drop if non-finite beyond threshold
                for v in self.view_names:
                    dfv = aligned[v]
                    if nonfinite_scope == 'numeric_only':
                        cols = [c for c in dfv.columns if pd.api.types.is_numeric_dtype(dfv[c])]
                        arr = dfv[cols].to_numpy(copy=False).astype(np.float32) if len(cols) else \
                              np.ones((len(dfv), 0), dtype=np.float32)
                    else:
                        # all columns numeric cast; non-numeric ignored in mask
                        cols = [c for c in dfv.columns if pd.api.types.is_numeric_dtype(dfv[c])]
                        arr = dfv[cols].to_numpy(copy=False).astype(np.float32) if len(cols) else \
                              np.ones((len(dfv), 0), dtype=np.float32)
                    if arr.size == 0:
                        continue
                    finite = np.isfinite(arr)
                    row_ok = (finite.sum(axis=1) >= (1.0 - self.max_row_na_frac) * finite.shape[1])
                    keep_rows &= row_ok
                if not keep_rows.all():
                    self.index = self.index[keep_rows]
                    aligned = {v: aligned[v].loc[self.index] for v in self.view_names}
                    if target is not None:
                        target = target.loc[self.index]

            else:
                # in-place imputation across views for numeric columns
                for v in self.view_names:
                    dfv = aligned[v]
                    df_num = dfv.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
                    if nonfinite_clean == 'impute_zero':
                        aligned[v][df_num.columns] = df_num.fillna(0.0)
                    elif nonfinite_clean == 'impute_mean':
                        aligned[v][df_num.columns] = df_num.fillna(df_num.mean(axis=0))

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
                std  = np.nanstd(num_vals, axis=0)
                vmin = np.nanmin(num_vals, axis=0)
                vmax = np.nanmax(num_vals, axis=0)
                eps_std = np.maximum(std, 1e-8)
                eps_rng = np.maximum(vmax - vmin, 1e-8)
            else:
                # No numeric cols
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

    # ------------------------------
    # Public API
    # ------------------------------

    def __len__(self) -> int:
        return self.number_of_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return one sample by aligned index (modulo N).

        Numeric path:
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
                    xd = np.where(np.isfinite(xd), xd, 0.0)  # one-hot mean ~ 0 for rare categories
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
                'index': index_value,
            }
            if self.target is not None:
                sample['target'] = _to_tensor(self.target[ridx].reshape(-1), self.dtype)
        else:
            sample = {
                'views': view_tensors,
                'masks': view_masks,
                'index': index_value,
            }
            if self.target is not None:
                sample['target'] = _to_tensor(self.target[ridx].reshape(-1), self.dtype)
        return sample

    def set_alpha(self, alpha: float, add_noise_in: Optional[str] = None):
        """Adjust jitter alpha (and optionally noise domain) at runtime."""
        self.alpha = float(alpha)
        if add_noise_in is not None:
            if add_noise_in not in {'raw', 'normalized', 'none'}:
                raise ValueError("add_noise_in must be 'raw', 'normalized', or 'none'")
            self.add_noise_in = add_noise_in

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
        if isinstance(normalization, (str, type(None))):
            modes = {v: normalization for v in self.view_names}
        elif isinstance(normalization, dict):
            modes = {v: normalization.get(v, '0mean') for v in self.view_names}
        else:
            raise TypeError("normalization must be str, dict, or None")
        for v, m in modes.items():
            if m not in {None, '01', '0mean'}:
                raise ValueError(f"Unsupported normalization mode '{m}' for view '{v}'")
        return modes

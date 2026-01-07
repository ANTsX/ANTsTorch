import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List


def pearson_multi(views_feats: List[torch.Tensor]) -> torch.Tensor:
    """
    Pearson alignment baseline over multiple views.

    For each unordered pair of views (i < j), this computes the per-feature
    Pearson cross-correlation between their projected latents and returns
    the **negative** mean of the diagonal correlations, averaged across
    all pairs. Minimizing this loss therefore maximizes per-dimension
    correlation across views.

    Parameters
    ----------
    views_feats : list[torch.Tensor]
        List of feature tensors, one per view, each of shape ``[B, D]``.
        All tensors must share the same batch size ``B`` and feature
        dimension ``D``. Float32 is recommended; inputs may reside on CPU
        or a CUDA device, but all tensors must be on the same device.

    Returns
    -------
    torch.Tensor
        A scalar (0-dim) tensor with the loss value:
        ``loss = - mean_{i<j} mean(diag(C_ij))``, where
        ``C_ij = X_i^T X_j / (B - 1)`` and each ``X`` is the standardized
        (zero-mean, unit-variance) version of the corresponding view’s
        features along the batch dimension. If fewer than two views are
        provided, returns ``tensor(0.0)`` on the inputs’ device/dtype.

    Notes
    -----
    * Each feature column is standardized over the batch with a small
      numerical floor (``eps = 1e-6``) on the standard deviation to avoid
      divide-by-zero.
    * Only the **diagonal** of the cross-correlation matrix is used
      (feature-aligned correlation). Redundancy between different feature
      dimensions is **not** penalized here; prefer Barlow Twins or VICReg
      if you want de-redundancy.
    * Fully differentiable; gradients flow into the inputs.
    * Computational cost is ``O(V^2 * B * D)`` for ``V`` views.

    Shape
    -----
    - Input: ``views_feats[v]`` is ``[B, D]`` for each view ``v``.
    - Output: scalar tensor ``[]``.

    Examples
    --------
    >>> x = torch.randn(64, 128)
    >>> y = x + 0.05 * torch.randn(64, 128)
    >>> loss = pearson_multi([x, y])
    >>> loss.shape
    torch.Size([])
    """
    V = len(views_feats)
    if V < 2:
        return torch.tensor(0.0, device=views_feats[0].device, dtype=views_feats[0].dtype)

    eps = 1e-6
    total = torch.tensor(0.0, device=views_feats[0].device, dtype=views_feats[0].dtype)
    n_pairs = 0
    for i in range(V):
        for j in range(i + 1, V):
            x = views_feats[i]
            y = views_feats[j]
            B = x.size(0)
            # standardize along batch
            x = (x - x.mean(dim=0)) / (x.std(dim=0) + eps)
            y = (y - y.mean(dim=0)) / (y.std(dim=0) + eps)
            C = (x.t() @ y) / max(B - 1, 1)  # [D,D]
            diag_mean = torch.diag(C).mean()
            total = total - diag_mean  # negative to maximize correlation
            n_pairs += 1
    return total / float(n_pairs) if n_pairs > 0 else total





def info_nce_multi(views_feats: List[torch.Tensor], T: float) -> torch.Tensor:
    """
    Multi-view InfoNCE / NT-Xent loss with multi-positives (SimCLR-style).

    For V aligned views of the same batch, this computes the NT-Xent
    contrastive loss where, for each anchor row, **positives are the
    same-sample rows from the other views**, and **all remaining rows in
    the concatenated batch are negatives**. Features are L2-normalized
    inside the function, cosine similarities are scaled by temperature
    ``T``, and a numerically-stable ``logsumexp`` is used for both the
    numerator (multi-positives) and denominator.

    Formally, with normalized features and similarities
    :math:`s_{ij} = \\langle x_i, x_j \\rangle / T`, the per-row loss is

    .. math::

        \\ell_i = -\\log
        \\frac{\\sum\\limits_{p \\in \\mathcal{P}(i)} \\exp(s_{ip})}
             {\\sum\\limits_{a \\neq i} \\exp(s_{ia})},

    where :math:`\\mathcal{P}(i)` indexes the **same sample** in all
    *other* views (multi-positive). The returned loss is the mean of
    :math:`\\ell_i` over all rows across all views.

    Parameters
    ----------
    views_feats : list[torch.Tensor]
        List of V feature tensors, one per view, each of shape ``[B, D]``.
        Rows must be **paired across views** (row k in every view refers
        to the same sample). ``V`` must be >= 2.
    T : float
        Temperature (scales cosine similarities). Internally clamped to
        ``max(T, 1e-8)`` for numerical stability.

    Returns
    -------
    torch.Tensor
        A scalar (0-dim) tensor with the InfoNCE/NT-Xent loss value,
        averaged over all rows in all views.

    Notes
    -----
    * Inputs are **L2-normalized** along ``dim=1`` inside the function, so
      you do not need to pre-normalize.
    * Self-similarities are removed from the denominator by masking.
    * Uses ``torch.logsumexp`` for stability with multiple positives.
    * Complexity is ``O((V·B)^2)`` due to the full similarity matrix.
    * Mixed precision is supported; loss math runs in fp32 once inputs are
      cast to float (recommend fp32 for alignment losses).

    Shape
    -----
    - Input: each ``views_feats[v]`` is ``[B, D]``.
    - Output: scalar tensor ``[]``.

    Examples
    --------
    >>> x = torch.randn(64, 128)                 # view 1
    >>> y = x + 0.05 * torch.randn(64, 128)      # view 2 (positives align by row)
    >>> loss = info_nce_multi([x, y], T=0.2)
    >>> loss.shape
    torch.Size([])

    References
    ----------
    .. [1] van den Oord, A., Li, Y., & Vinyals, O. (2018).
           *Representation Learning with Contrastive Predictive Coding*.
           arXiv:1807.03748.
    .. [2] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020).
           *A Simple Framework for Contrastive Learning of Visual Representations*
           (SimCLR). ICML 2020.
    """
    ...
    feats = [nn.functional.normalize(f, dim=1) for f in views_feats]
    B = feats[0].size(0)
    V = len(feats)
    X = torch.cat(feats, dim=0)               # [V*B, D]
    sim = X @ X.t() / max(T, 1e-8)            # [VB, VB]
    mask = torch.eye(V*B, device=X.device, dtype=torch.bool)
    sim.masked_fill_(mask, -1e9)              # remove self

    ids = torch.arange(B, device=X.device).repeat(V)  # [VB]
    loss_rows = []
    arange_vb = torch.arange(V*B, device=X.device)
    for i in range(V*B):
        pos = (ids == ids[i]) & (arange_vb != i)
        denom = torch.logsumexp(sim[i], dim=0)
        numer = torch.logsumexp(sim[i][pos], dim=0)
        loss_rows.append(-(numer - denom))
    return torch.stack(loss_rows, dim=0).mean()




def barlow_twins_multi(views_feats: List[torch.Tensor], lam: float = 5e-3) -> torch.Tensor:
    """
    Multi-view Barlow Twins alignment loss.

    For each unordered pair of views (i < j), this computes the
    cross-correlation matrix between **standardized** features and
    penalizes deviation from the identity matrix:

    .. math::

        \\mathcal{L}_{\\text{BT}}(i,j)
        = \\sum_k (C_{kk} - 1)^2
        \\, + \\, \\lambda \\sum_{k\\ne\\ell} C_{k\\ell}^2,

    where :math:`C = X^\\top Y / (B-1)`, and :math:`X, Y \\in \\mathbb{R}^{B\\times D}`
    are the per-batch **zero-mean / unit-variance** versions of the two
    views' features. The function returns the average of
    :math:`\\mathcal{L}_{\\text{BT}}(i,j)` over all view pairs.

    Intuition: the diagonal term encourages **featurewise agreement**
    across views (corr ≈ 1), while the off-diagonal term discourages
    **redundancy** (features become decorrelated / whitened).

    Parameters
    ----------
    views_feats : list[torch.Tensor]
        List of feature tensors, one per view, each of shape ``[B, D]``.
        Rows must be aligned across views (row k corresponds to the same
        sample). All tensors must share the same ``B`` and ``D`` and live
        on the same device. Float32 is recommended for numerical stability.
    lam : float, default=5e-3
        Weight on the off-diagonal penalty. Larger values enforce stronger
        de-redundancy; smaller values emphasize diagonal ≈ 1.

    Returns
    -------
    torch.Tensor
        A scalar (0-dim) tensor with the Barlow Twins loss value, averaged
        over all unordered view pairs.

    Notes
    -----
    * Each feature column is standardized over the batch with a small
      variance floor (``eps``) internally to avoid divide-by-zero.
    * No negatives are used; works well with moderate batch sizes.
      (Empirically more stable with ``B ≥ 64`` for covariance estimates.)
    * Fully differentiable; gradients flow into all inputs.
    * Computational cost is ``O(V^2 · B · D)`` for ``V`` views.

    Shape
    -----
    - Input: ``views_feats[v]`` is ``[B, D]`` for each view ``v``.
    - Output: scalar tensor ``[]``.

    Examples
    --------
    >>> x = torch.randn(64, 256)                 # view 1
    >>> y = x + 0.05 * torch.randn(64, 256)      # view 2 (paired)
    >>> loss = barlow_twins_multi([x, y], lam=5e-3)
    >>> loss.shape
    torch.Size([])

    References
    ----------
    .. [1] Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021).
           *Barlow Twins: Self-Supervised Learning via Redundancy Reduction.*
           ICML 2021.
    """
    B = views_feats[0].size(0)
    losses = []
    eye_cache = None
    for i in range(len(views_feats)):
        Zi = (views_feats[i] - views_feats[i].mean(0)) / (views_feats[i].std(0) + 1e-5)
        Zi = torch.nan_to_num(Zi)
        for j in range(i+1, len(views_feats)):
            Zi = torch.nan_to_num(Zi)
            Zj = (views_feats[j] - views_feats[j].mean(0)) / (views_feats[j].std(0) + 1e-5)
            Zj = torch.nan_to_num(Zj)
            C = (Zi.t() @ Zj) / max(B, 1)
            C = torch.nan_to_num(C)
            if eye_cache is None or eye_cache.size(0) != C.size(0):
                eye_cache = torch.eye(C.size(0), device=C.device, dtype=C.dtype)
            on = (C.diag() - 1).pow(2).sum()
            off = (C - eye_cache).pow(2).sum() - (C.diag() - 1).pow(2).sum()
            losses.append(on + lam * off)
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=views_feats[0].device)




def vicreg_multi(views_feats: List[torch.Tensor],
                 w_inv: float = 25.0,
                 w_var: float = 25.0,
                 w_cov: float = 1.0,
                 gamma: float = 1.0) -> torch.Tensor:
    """
    Multi-view VICReg (Variance–Invariance–Covariance Regularization) loss.

    Extends VICReg to V aligned views of the same batch. For each unordered
    pair of views (i < j), the **invariance** term pulls paired samples
    together via MSE in the projector space. For each individual view,
    the **variance** term enforces a per-feature standard deviation floor
    to prevent collapse, and the **covariance** term penalizes off-diagonal
    entries of the (centered) feature covariance to reduce redundancy.

    The returned loss is:
    ``w_inv * mean_pair(MSE) + w_var * mean_view(VarFloor) + w_cov * mean_view(CovOffDiag)``.

    Formally, let ``X ∈ R^{B×D}`` denote a view's features centered along the
    batch, and ``std_j`` the per-feature standard deviation with a small
    numerical floor (``eps``). Then

    .. math::

        \\mathcal{L}_{\\text{inv}} =
        \\frac{1}{|\\mathcal{P}|} \\sum_{(i,j)\\in\\mathcal{P}}
        \\frac{1}{B D} \\lVert X^{(i)} - X^{(j)} \\rVert_2^2,

        \\quad
        \\mathcal{L}_{\\text{var}} =
        \\frac{1}{V D} \\sum_{v=1}^V \\sum_{j=1}^D
        \\big[\\max(0, \\gamma - \\operatorname{std}_j(X^{(v)}))\\big]^2,

        \\quad
        \\mathcal{L}_{\\text{cov}} =
        \\frac{1}{V} \\sum_{v=1}^V
        \\frac{1}{D} \\sum_{k\\neq \\ell}
        \\big( \\operatorname{Cov}(X^{(v)})_{k\\ell} \\big)^2,

    where ``Cov(X) = X^T X / (B-1)`` after centering.

    Parameters
    ----------
    views_feats : list[torch.Tensor]
        List of V feature tensors, one per view, each of shape ``[B, D]``.
        Rows must be paired across views (row k is the same sample). All
        tensors must share the same ``B`` and ``D`` and live on the same device.
        Float32 is recommended for numerical stability.
    w_inv : float, default=25.0
        Weight of the invariance (pairwise MSE) term.
    w_var : float, default=25.0
        Weight of the variance floor term.
    w_cov : float, default=1.0
        Weight of the covariance off-diagonal penalty.
    gamma : float, default=1.0
        Target minimum standard deviation per feature (variance floor).
        Typical choice is ``1.0`` when projector outputs are roughly
        standardized; tune if your projector uses different scaling.

    Returns
    -------
    torch.Tensor
        A scalar (0-dim) tensor with the VICReg loss value aggregated over
        all unordered view pairs (for the invariance term) and over views
        (for the variance & covariance terms).

    Notes
    -----
    * No negatives are required; VICReg is robust with modest batch sizes.
    * We recommend computing this loss in **fp32** (cast inside the function)
      and adding a small ``eps`` (e.g., ``1e-4``) to standard deviations and
      denominators to avoid divide-by-zero.
    * The invariance part averages over all unordered view pairs; variance
      and covariance parts average over views.
    * Computational complexity is ``O(V^2 · B · D)`` for the pairwise MSE
      plus ``O(V · B · D)`` for statistics.

    Shape
    -----
    - Input: each ``views_feats[v]`` is ``[B, D]``.
    - Output: scalar tensor ``[]``.

    Examples
    --------
    >>> x = torch.randn(64, 256)                         # view 1
    >>> y = x + 0.05 * torch.randn(64, 256)              # view 2 (paired)
    >>> loss = vicreg_multi([x, y], w_inv=25, w_var=25, w_cov=1, gamma=1.0)
    >>> loss.shape
    torch.Size([])

    References
    ----------
    .. [1] Bardes, A., Ponce, J., & LeCun, Y. (2021).
           *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning*.
           arXiv:2105.04906.
    """

    def _offdiag(x: torch.Tensor) -> torch.Tensor:
        # return the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    V = len(views_feats)
    B = views_feats[0].size(0)
    eps = 1e-4

    # Invariance (pairwise MSE)
    inv = torch.tensor(0.0, device=views_feats[0].device, dtype=views_feats[0].dtype)
    n_pairs = 0
    for i in range(V):
        for j in range(i + 1, V):
            inv = inv + F.mse_loss(views_feats[i], views_feats[j])
            n_pairs += 1
    if n_pairs > 0:
        inv = inv / float(n_pairs)

    # Variance & Covariance (per view)
    var_acc = torch.tensor(0.0, device=views_feats[0].device, dtype=views_feats[0].dtype)
    cov_acc = torch.tensor(0.0, device=views_feats[0].device, dtype=views_feats[0].dtype)
    for v in range(V):
        z = views_feats[v]
        # variance
        std = z.std(dim=0) + eps
        var = torch.relu(gamma - std).pow(2).mean()
        var_acc = var_acc + var
        # covariance
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.t() @ zc) / max(B - 1, 1)
        cov = _offdiag(cov).pow(2).sum() / cov.size(0)
        cov_acc = cov_acc + cov

    var_acc = var_acc / float(V)
    cov_acc = cov_acc / float(V)

    return w_inv * inv + w_var * var_acc + w_cov * cov_acc




def hsic_biased(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma_x: float = 0.0,
    sigma_y: float = 0.0,
) -> torch.Tensor:
    """
    Biased HSIC between two views using RBF kernels.

    Computes the (biased) Hilbert–Schmidt Independence Criterion (HSIC)
    between two batches of features ``x`` and ``y``. RBF kernels are
    formed on each view with bandwidths ``sigma_x`` and ``sigma_y``; the
    centered Gram matrices are then combined as:

    .. math::

        \\operatorname{HSIC}_b(x,y)
        \\,=\\, \\frac{1}{(B-1)^2} \\, \\operatorname{tr}( K H L H ),

    where :math:`K` and :math:`L` are the RBF Gram matrices for ``x`` and
    ``y`` respectively, :math:`H = I - \\tfrac{1}{B}\\mathbf{1}\\mathbf{1}^\\top`
    is the centering matrix, and ``B`` is the batch size.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape ``[B, D_x]`` (batch of features for view X).
    y : torch.Tensor
        Tensor of shape ``[B, D_y]`` (batch of features for view Y).
        ``x`` and ``y`` must have the same batch size ``B`` and reside on
        the same device; float32 is recommended for numerical stability.
    sigma_x : float, default=0.0
        RBF bandwidth for ``x``. If ``0.0``, a **median heuristic** is
        used (median pairwise distance on the batch, with small numerical
        flooring if needed).
    sigma_y : float, default=0.0
        RBF bandwidth for ``y``. If ``0.0``, uses the median heuristic.

    Returns
    -------
    torch.Tensor
        A non-negative scalar (0-dim) tensor with the biased HSIC value.
        Larger values indicate stronger statistical dependence between
        ``x`` and ``y`` over the batch.

    Notes
    -----
    * This is the **biased** estimator (uses all entries including the
      diagonal after centering). It is simple, low-variance, and works
      well in practice. If you require the **unbiased** U-statistic
      version, implement the corresponding correction on the Gram
      matrices before the trace.
    * Bandwidth selection strongly affects the magnitude of HSIC; the
      median heuristic (``sigma=0``) is a good default. You can also
      pass explicit ``sigma_x/sigma_y`` if prior scale knowledge exists.
    * Computational cost is ``O(B^2 · (D_x + D_y))`` to form the two Gram
      matrices and center them.

    Shape
    -----
    - Input: ``x: [B, D_x]``, ``y: [B, D_y]`` (same ``B``).
    - Output: scalar tensor ``[]``.

    Examples
    --------
    >>> B, Dx, Dy = 128, 16, 16
    >>> x = torch.randn(B, Dx)
    >>> y_indep = torch.randn(B, Dy)
    >>> y_dep = x + 0.2 * torch.randn(B, Dy)
    >>> hs_ind = hsic_biased(x, y_indep, sigma_x=0.0, sigma_y=0.0)
    >>> hs_dep = hsic_biased(x, y_dep,   sigma_x=0.0, sigma_y=0.0)
    >>> (hs_dep > hs_ind).item()
    True

    References
    ----------
    .. [1] Gretton, A., Bousquet, O., Smola, A., & Schölkopf, B. (2005).
           *Measuring Statistical Dependence with Hilbert-Schmidt Norms*.
           ALT 2005.
    .. [2] Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Schölkopf, B., & Smola, A. (2008).
           *A Kernel Statistical Test of Independence*. NIPS 2008.
    """

    def _pairwise_sq_dists(x: torch.Tensor) -> torch.Tensor:
        # x: [B,D]
        x_norm = (x * x).sum(dim=1, keepdim=True)  # [B,1]
        d2 = x_norm + x_norm.t() - 2.0 * (x @ x.t())
        d2 = torch.clamp(d2, min=0.0)
        return d2

    def _median_heuristic_sigma(x: torch.Tensor) -> float:
        # median of upper-triangular pairwise distances (sqrt of squared distances)
        with torch.no_grad():
            d2 = _pairwise_sq_dists(x)
            triu = d2.triu(diagonal=1)
            vals = triu[triu > 0].flatten()
            if vals.numel() == 0:
                return 1.0
            med = torch.median(torch.sqrt(vals)).item()
            if med <= 0 or not (med == med):  # NaN check
                return 1.0
            return med

    def _rbf_gram(x: torch.Tensor, sigma: float) -> torch.Tensor:
        d2 = _pairwise_sq_dists(x)
        if sigma <= 0:
            sigma = _median_heuristic_sigma(x)
        gamma = 1.0 / (2.0 * (sigma ** 2) + 1e-12)
        K = torch.exp(-gamma * d2)
        return K

    assert x.size(0) == y.size(0), "Batch sizes must match for HSIC."
    n = x.size(0)
    if n < 2:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    K = _rbf_gram(x, sigma_x)
    L = _rbf_gram(y, sigma_y)

    I = torch.eye(n, device=x.device, dtype=x.dtype)
    H = I - (1.0 / n) * torch.ones_like(I)
    KH = K @ H
    HLH = H @ L @ H
    hsic = torch.trace(KH @ HLH) / max((n - 1) ** 2, 1)
    return hsic

def hsic_multi(views_feats: List[torch.Tensor], sigma: float = 0.0) -> torch.Tensor:
    """
    Average negative HSIC (so it's a loss) over all (i<j) view pairs.
    If sigma==0, uses median heuristic per view for the RBF bandwidths.
    """
    V = len(views_feats)
    if V < 2:
        return torch.tensor(0.0, device=views_feats[0].device, dtype=views_feats[0].dtype)

    loss = torch.tensor(0.0, device=views_feats[0].device, dtype=views_feats[0].dtype)
    n_pairs = 0
    for i in range(V):
        for j in range(i + 1, V):
            # separate bandwidths for each view improve stability
            sig_i = sigma if sigma > 0 else 0.0
            sig_j = sigma if sigma > 0 else 0.0
            hs = hsic_biased(views_feats[i], views_feats[j], sigma_x=sig_i, sigma_y=sig_j)
            loss = loss - hs  # negative to maximize dependence
            n_pairs += 1
    if n_pairs > 0:
        loss = loss / float(n_pairs)
    return loss


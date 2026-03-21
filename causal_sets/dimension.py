"""
Dimension estimators for causal sets.

Multiple independent methods for extracting the effective dimension of a causal set,
which should recover the manifold dimension for sprinkled causets and can reveal
dimensional flow in dynamical causets.
"""

import numpy as np
from math import gamma, lgamma
from .core import CausalSet


def myrheim_meyer(causet: CausalSet) -> float:
    """
    Myrheim-Meyer dimension estimator.

    For a causet sprinkled into d-dimensional Minkowski spacetime, the ordering
    fraction f (fraction of causally related pairs) is related to the dimension by:

        f = Gamma(d+1) * Gamma(d/2) / (4 * Gamma(3d/2))

    for a causal diamond (Alexandrov interval). This can be inverted numerically
    to extract d from the measured f.

    Reference: Myrheim (1978), Meyer (1988), Reid (2003).
    """
    f_unordered = causet.ordering_fraction()  # R / C(n,2)
    if f_unordered <= 0 or f_unordered >= 1:
        return float('nan')

    # The MM formula gives f_d = R / (n(n-1)) = (1/2) * R / C(n,2)
    # So f_d = f_unordered / 2
    f = f_unordered / 2.0

    # Numerically invert the f(d) relation
    # f(d) = Gamma(d+1) * Gamma(d/2) / (4 * Gamma(3d/2))
    return _invert_ordering_fraction(f)


def _ordering_fraction_theory(d: float) -> float:
    """Theoretical ordering fraction for d-dimensional Minkowski causal diamond."""
    if d <= 0:
        return 1.0
    try:
        log_f = lgamma(d + 1) + lgamma(d / 2) - np.log(4) - lgamma(3 * d / 2)
        return np.exp(log_f)
    except (ValueError, OverflowError):
        return 0.0


def _invert_ordering_fraction(f: float, tol: float = 1e-6) -> float:
    """Numerically invert f(d) to find dimension from ordering fraction."""
    # Binary search on d in [1, 20]
    d_low, d_high = 0.5, 20.0
    for _ in range(100):
        d_mid = (d_low + d_high) / 2
        f_mid = _ordering_fraction_theory(d_mid)
        if f_mid > f:
            d_low = d_mid
        else:
            d_high = d_mid
        if d_high - d_low < tol:
            break
    return (d_low + d_high) / 2


def interval_dimension(causet: CausalSet, sample_size: int = 1000,
                       rng: np.random.Generator = None) -> float:
    """
    Estimate dimension from the distribution of interval sizes.

    For a causet sprinkled into d-dimensional Minkowski spacetime, the expected
    number of elements in an interval [x, y] of volume V is:

        <n> = rho * V * Gamma(d+1) * Gamma(d/2) / (4 * Gamma(3d/2))

    The ratio <n^2>/<n>^2 depends only on d, allowing dimension extraction.

    This uses the relation:
        <C_k> / <C_1>^k = f_d(k)
    where C_k is the k-chain abundance (number of k-element chains in an interval).

    Simplified version: use the ratio of mean to variance of interval counts.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample random related pairs and compute interval sizes
    related_pairs = []
    for i in range(causet.n):
        for j in range(i + 1, causet.n):
            if causet.order[i, j]:
                related_pairs.append((i, j))

    if len(related_pairs) < 10:
        return float('nan')

    # Sample if too many pairs
    if len(related_pairs) > sample_size:
        indices = rng.choice(len(related_pairs), sample_size, replace=False)
        related_pairs = [related_pairs[idx] for idx in indices]

    intervals = []
    for i, j in related_pairs:
        intervals.append(causet.interval_count(i, j))

    intervals = np.array(intervals, dtype=float)
    mean_n = np.mean(intervals)
    if mean_n < 0.1:
        return float('nan')

    # For a d-dimensional Alexandrov interval with n elements:
    # The fraction of 2-chains relative to pairs gives dimension info
    # Use the simple Myrheim-Meyer on the sub-intervals
    # Actually, let's use the mean interval size relative to total size
    # as a cross-check on Myrheim-Meyer
    mean_interval = np.mean(intervals)
    var_interval = np.var(intervals)

    if mean_interval <= 0:
        return float('nan')

    # The ratio var/mean for a Poisson process in d dimensions
    # scales in a dimension-dependent way
    ratio = var_interval / mean_interval if mean_interval > 0 else float('nan')
    return ratio  # Return raw ratio; interpretation requires calibration


def chain_length_distribution(causet: CausalSet, max_length: int = 50) -> np.ndarray:
    """
    Compute the distribution of maximal chain lengths from each element.

    For a causet faithfully embedded in d-dimensional Minkowski spacetime,
    the longest chain through N elements scales as N^(1/d), providing
    another dimension estimator.

    Returns array where dist[k] = number of elements whose longest forward chain has length k.
    """
    n = causet.n
    # dp[i] = length of longest chain starting from element i (going forward)
    dp = np.ones(n, dtype=int)

    # Process in reverse order
    for i in range(n - 2, -1, -1):
        max_forward = 0
        for j in range(i + 1, n):
            if causet.order[i, j]:
                max_forward = max(max_forward, dp[j])
        dp[i] = 1 + max_forward

    # Build distribution
    dist = np.zeros(max_length, dtype=int)
    for length in dp:
        if length < max_length:
            dist[length] += 1

    return dist


def spectral_dimension(causet: CausalSet, sigma_range: tuple = (0.1, 10.0),
                       n_sigma: int = 50) -> tuple:
    """
    Estimate the spectral dimension via a diffusion process on the causet.

    The spectral dimension d_s is defined via the return probability of a
    random walk:
        P(sigma) ~ sigma^(-d_s/2)

    where sigma is the diffusion time.

    On a causal set, we use the link matrix as the adjacency matrix for
    the diffusion process. The spectral dimension is:
        d_s(sigma) = -2 * d(ln P(sigma)) / d(ln sigma)

    Returns (sigma_values, d_s_values).
    """
    links = causet.link_matrix()

    # Build the graph Laplacian from the link matrix (symmetrized)
    adj = links + links.T  # undirected version
    degree = np.sum(adj, axis=1)

    if np.any(degree == 0):
        # Remove isolated nodes for Laplacian
        mask = degree > 0
        adj = adj[np.ix_(mask, mask)]
        degree = degree[mask]

    n = adj.shape[0]
    if n < 3:
        return np.array([]), np.array([])

    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    laplacian = np.eye(n) - d_inv_sqrt @ adj @ d_inv_sqrt

    # Eigendecompose
    eigenvalues = np.linalg.eigvalsh(laplacian)
    eigenvalues = np.clip(eigenvalues, 0, None)  # numerical stability

    # Return probability: P(sigma) = (1/n) * sum_k exp(-lambda_k * sigma)
    sigmas = np.logspace(np.log10(sigma_range[0]), np.log10(sigma_range[1]), n_sigma)
    P = np.zeros(n_sigma)
    for idx, sigma in enumerate(sigmas):
        P[idx] = np.mean(np.exp(-eigenvalues * sigma))

    # Spectral dimension: d_s = -2 * d(ln P)/d(ln sigma)
    ln_P = np.log(P + 1e-300)
    ln_sigma = np.log(sigmas)
    d_s = -2 * np.gradient(ln_P, ln_sigma)

    return sigmas, d_s

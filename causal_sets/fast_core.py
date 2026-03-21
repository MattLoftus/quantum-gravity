"""
Optimized causal set operations using vectorized NumPy.
Replaces the O(n^3) loops in core.py with matrix operations.
"""

import numpy as np
from typing import Optional


class FastCausalSet:
    """Causal set with vectorized operations for larger sizes (N ~ 1000-5000)."""

    def __init__(self, n: int = 0):
        self.n = n
        self.order = np.zeros((n, n), dtype=np.bool_) if n > 0 else np.zeros((0, 0), dtype=np.bool_)

    @property
    def size(self) -> int:
        return self.n

    def num_relations(self) -> int:
        return int(np.sum(self.order))

    def ordering_fraction(self) -> float:
        if self.n < 2:
            return 0.0
        return self.num_relations() / (self.n * (self.n - 1) / 2)

    def link_matrix(self) -> np.ndarray:
        """Compute link matrix using matrix operations.
        L[i,j] = 1 if i < j and no k with i < k < j.
        Equivalent to: links = order AND NOT (order @ order)
        """
        # order^2[i,j] = number of paths of length 2 from i to j
        # If order^2[i,j] > 0, then (i,j) is NOT a link
        path2 = self.order.astype(np.int32) @ self.order.astype(np.int32)
        links = self.order & (path2 == 0)
        return links

    def longest_chain(self) -> int:
        if self.n == 0:
            return 0
        dp = np.ones(self.n, dtype=int)
        for j in range(self.n):
            predecessors = np.where(self.order[:j, j])[0]
            if len(predecessors) > 0:
                dp[j] = np.max(dp[predecessors]) + 1
        return int(np.max(dp))

    def interval_sizes_vectorized(self) -> tuple:
        """
        For all related pairs (i,j), compute the interval size |I(i,j)|.
        Returns (pair_indices, sizes) where pair_indices is (K, 2) array.

        Uses the formula: for each related pair (i,j),
        |I(i,j)| = number of k where order[i,k] AND order[k,j].
        This is exactly (order^T @ order)[i,j] restricted to related pairs.
        Wait no: it's sum_k order[i,k] * order[k,j] = (order @ order)[i,j].
        """
        # Matrix product gives interval sizes for all pairs
        interval_matrix = self.order.astype(np.int32) @ self.order.astype(np.int32)

        # Get related pairs
        i_idx, j_idx = np.where(np.triu(self.order, k=1))
        sizes = interval_matrix[i_idx, j_idx]

        return np.column_stack([i_idx, j_idx]), sizes


def sprinkle_fast(n: int, dim: int = 2, extent_t: float = 1.0,
                  region: str = 'diamond',
                  rng: Optional[np.random.Generator] = None) -> tuple:
    """Fast vectorized sprinkling into Minkowski causal diamond."""
    if rng is None:
        rng = np.random.default_rng()

    if region == 'diamond':
        coords_list = []
        while len(coords_list) < n:
            batch = n * 4
            candidates = np.zeros((batch, dim))
            candidates[:, 0] = rng.uniform(-extent_t, extent_t, batch)
            for d in range(1, dim):
                candidates[:, d] = rng.uniform(-extent_t, extent_t, batch)
            r = np.sqrt(np.sum(candidates[:, 1:] ** 2, axis=1))
            mask = np.abs(candidates[:, 0]) + r <= extent_t
            accepted = candidates[mask]
            coords_list.append(accepted)
        coords = np.vstack(coords_list)[:n]
    else:
        coords = np.zeros((n, dim))
        coords[:, 0] = rng.uniform(0, extent_t, n)
        for d in range(1, dim):
            coords[:, d] = rng.uniform(-extent_t, extent_t, n)

    # Sort by time
    coords = coords[np.argsort(coords[:, 0])]

    # Vectorized causal order construction
    causet = FastCausalSet(n)

    # For each pair (i, j) with i < j (already time-ordered):
    # related if dt^2 >= sum(dx_k^2)
    # Vectorized: compute all pairwise differences
    for i in range(n):
        if i == n - 1:
            break
        dt = coords[i + 1:, 0] - coords[i, 0]  # all positive (time-ordered)
        dx_sq = np.sum((coords[i + 1:, 1:] - coords[i, 1:]) ** 2, axis=1)
        related = dt * dt >= dx_sq
        causet.order[i, i + 1:] = related

    return causet, coords


def csg_fast(n: int, coupling: float = 0.0,
             rng: Optional[np.random.Generator] = None) -> FastCausalSet:
    """Fast CSG transitive percolation."""
    if rng is None:
        rng = np.random.default_rng()

    causet = FastCausalSet(n)

    for new in range(1, n):
        # Each existing element independently precedes the new one with prob p
        random_vals = rng.random(new)
        direct = random_vals < coupling

        # Transitive closure: if ancestor < old and old < new, then ancestor < new
        past_of_new = direct.copy()
        for old in range(new):
            if past_of_new[old]:
                past_of_new[:old] |= causet.order[:old, old]

        causet.order[:new, new] = past_of_new

    return causet


def spectral_dimension_fast(causet: FastCausalSet,
                            sigma_range: tuple = (0.01, 100.0),
                            n_sigma: int = 60) -> tuple:
    """Spectral dimension via diffusion on the link graph."""
    links = causet.link_matrix()
    adj = links | links.T  # symmetrize
    degree = np.sum(adj, axis=1).astype(float)

    # Remove isolated nodes
    mask = degree > 0
    adj = adj[np.ix_(mask, mask)].astype(float)
    degree = degree[mask]
    n = adj.shape[0]

    if n < 5:
        return np.array([]), np.array([])

    # Normalized Laplacian
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    laplacian = np.eye(n) - d_inv_sqrt @ adj @ d_inv_sqrt
    eigenvalues = np.linalg.eigvalsh(laplacian)
    eigenvalues = np.clip(eigenvalues, 0, None)

    sigmas = np.logspace(np.log10(sigma_range[0]), np.log10(sigma_range[1]), n_sigma)
    P = np.zeros(n_sigma)
    for idx, sigma in enumerate(sigmas):
        P[idx] = np.mean(np.exp(-eigenvalues * sigma))

    ln_P = np.log(P + 1e-300)
    ln_sigma = np.log(sigmas)
    d_s = -2 * np.gradient(ln_P, ln_sigma)

    return sigmas, d_s

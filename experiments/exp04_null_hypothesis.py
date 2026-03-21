"""
Experiment 04: Null hypothesis tests.

Is the d_s ≈ 2 feature in CSG causets meaningful, or does any graph
show d_s ≈ 2 at some scale?

Compare CSG causets against:
1. Erdos-Renyi random graphs (no causal structure)
2. Random DAGs (directed acyclic graphs — partial order without the
   specific CSG dynamics)
3. Regular lattices

If d_s ≈ 2 appears in ALL graph types, it's an artifact.
If it appears only in CSG causets (and sprinkled causets), it's physical.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from causal_sets.fast_core import FastCausalSet, csg_fast, sprinkle_fast, spectral_dimension_fast


def random_dag(n: int, edge_prob: float = 0.1,
               rng: np.random.Generator = None) -> FastCausalSet:
    """Random DAG: for each pair (i,j) with i<j, add edge with probability p.
    Then take transitive closure. No causal/physical structure."""
    if rng is None:
        rng = np.random.default_rng()

    cs = FastCausalSet(n)
    # Random upper triangular edges
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < edge_prob:
                cs.order[i, j] = True

    # Transitive closure
    changed = True
    while changed:
        new_order = cs.order | (cs.order.astype(np.int8) @ cs.order.astype(np.int8) > 0)
        changed = np.any(new_order != cs.order)
        cs.order = new_order

    return cs


def erdos_renyi_undirected(n: int, edge_prob: float = 0.05,
                            rng: np.random.Generator = None) -> np.ndarray:
    """Return adjacency matrix (symmetric) for ER graph. Not a causet."""
    if rng is None:
        rng = np.random.default_rng()
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < edge_prob:
                adj[i, j] = True
                adj[j, i] = True
    return adj


def spectral_dim_from_adj(adj: np.ndarray, sigma_range=(0.005, 200.0), n_sigma=60):
    """Compute spectral dimension from an adjacency matrix."""
    degree = np.sum(adj, axis=1).astype(float)
    mask = degree > 0
    adj = adj[np.ix_(mask, mask)].astype(float)
    degree = degree[mask]
    n = adj.shape[0]
    if n < 5:
        return np.array([]), np.array([])

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


def lattice_1d(n: int) -> np.ndarray:
    """1D chain graph."""
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n - 1):
        adj[i, i + 1] = True
        adj[i + 1, i] = True
    return adj


def lattice_2d(side: int) -> np.ndarray:
    """2D square lattice."""
    n = side * side
    adj = np.zeros((n, n), dtype=bool)
    for i in range(side):
        for j in range(side):
            idx = i * side + j
            if j + 1 < side:
                adj[idx, idx + 1] = True
                adj[idx + 1, idx] = True
            if i + 1 < side:
                adj[idx, idx + side] = True
                adj[idx + side, idx] = True
    return adj


def main():
    rng = np.random.default_rng(42)
    n = 500

    print("=" * 70)
    print("EXPERIMENT 04: Null Hypothesis — Is d_s≈2 Universal?")
    print("=" * 70)

    results = {}

    # 1. CSG causets at different couplings
    print("\n--- CSG Causets (link graph) ---")
    for p in [0.05, 0.10, 0.20]:
        cs = csg_fast(n, coupling=p, rng=rng)
        sigmas, d_s = spectral_dimension_fast(cs, sigma_range=(0.005, 200.0), n_sigma=60)
        if len(sigmas) > 0:
            peak_idx = np.argmax(d_s)
            closest_2 = np.argmin(np.abs(d_s - 2.0))
            results[f'CSG p={p}'] = (d_s[peak_idx], sigmas[closest_2])
            print(f"  p={p:.2f}: peak d_s={d_s[peak_idx]:.2f}, d_s=2 at sigma={sigmas[closest_2]:.3f}")

    # 2. Sprinkled 2D and 4D causets
    print("\n--- Sprinkled Causets (link graph) ---")
    for dim in [2, 4]:
        cs, _ = sprinkle_fast(n, dim=dim, extent_t=1.0, region='diamond', rng=rng)
        sigmas, d_s = spectral_dimension_fast(cs, sigma_range=(0.005, 200.0), n_sigma=60)
        if len(sigmas) > 0:
            peak_idx = np.argmax(d_s)
            closest_2 = np.argmin(np.abs(d_s - 2.0))
            results[f'Sprinkle d={dim}'] = (d_s[peak_idx], sigmas[closest_2])
            print(f"  d={dim}: peak d_s={d_s[peak_idx]:.2f}, d_s=2 at sigma={sigmas[closest_2]:.3f}")

    # 3. Random DAGs
    print("\n--- Random DAGs (link graph) ---")
    for ep in [0.01, 0.05, 0.10]:
        dag = random_dag(n, edge_prob=ep, rng=rng)
        sigmas, d_s = spectral_dimension_fast(dag, sigma_range=(0.005, 200.0), n_sigma=60)
        if len(sigmas) > 0:
            peak_idx = np.argmax(d_s)
            closest_2 = np.argmin(np.abs(d_s - 2.0))
            results[f'DAG p={ep}'] = (d_s[peak_idx], sigmas[closest_2])
            print(f"  p={ep:.2f}: peak d_s={d_s[peak_idx]:.2f}, d_s=2 at sigma={sigmas[closest_2]:.3f}")

    # 4. Erdos-Renyi random graphs
    print("\n--- Erdos-Renyi Random Graphs ---")
    for ep in [0.01, 0.05, 0.10]:
        adj = erdos_renyi_undirected(n, edge_prob=ep, rng=rng)
        sigmas, d_s = spectral_dim_from_adj(adj, sigma_range=(0.005, 200.0), n_sigma=60)
        if len(sigmas) > 0:
            peak_idx = np.argmax(d_s)
            closest_2 = np.argmin(np.abs(d_s - 2.0))
            results[f'ER p={ep}'] = (d_s[peak_idx], sigmas[closest_2])
            print(f"  p={ep:.2f}: peak d_s={d_s[peak_idx]:.2f}, d_s=2 at sigma={sigmas[closest_2]:.3f}")

    # 5. Regular lattices
    print("\n--- Regular Lattices ---")
    adj_1d = lattice_1d(n)
    sigmas, d_s = spectral_dim_from_adj(adj_1d, sigma_range=(0.005, 200.0), n_sigma=60)
    if len(sigmas) > 0:
        peak_idx = np.argmax(d_s)
        closest_2 = np.argmin(np.abs(d_s - 2.0))
        results['Lattice 1D'] = (d_s[peak_idx], sigmas[closest_2])
        print(f"  1D chain (N={n}): peak d_s={d_s[peak_idx]:.2f}, d_s=2 at sigma={sigmas[closest_2]:.3f}")

    side = int(np.sqrt(n))
    adj_2d = lattice_2d(side)
    sigmas, d_s = spectral_dim_from_adj(adj_2d, sigma_range=(0.005, 200.0), n_sigma=60)
    if len(sigmas) > 0:
        peak_idx = np.argmax(d_s)
        closest_2 = np.argmin(np.abs(d_s - 2.0))
        n_2d = side * side
        results['Lattice 2D'] = (d_s[peak_idx], sigmas[closest_2])
        print(f"  2D grid ({side}x{side}={n_2d}): peak d_s={d_s[peak_idx]:.2f}, d_s=2 at sigma={sigmas[closest_2]:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Is d_s ≈ 2 Universal?")
    print("=" * 70)
    print(f"{'Model':>25} {'Peak d_s':>10} {'σ where d_s=2':>15}")
    print("-" * 55)
    for name, (peak, sigma_2) in results.items():
        print(f"{name:>25} {peak:>10.2f} {sigma_2:>15.3f}")

    print("\nConclusion: if ALL models show d_s≈2 at SOME scale, it's a")
    print("generic graph property. If only causal/manifold graphs show it,")
    print("it may be physically meaningful.")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()

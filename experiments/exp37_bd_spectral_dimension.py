"""
Experiment 37: Spectral Dimension from the BD d'Alembertian

The link-graph Laplacian gives d_s ≈ 5 on sprinkled 2D causets (wrong).
The BD d'Alembertian is designed to approximate the continuum wave operator.
Does it give d_s ≈ 2 (correct)?

The BD d'Alembertian on a causal set (2D, non-local with parameter epsilon):

  B_eps phi(x) = (4*eps/l^2) * [phi(x) - 2*eps * sum_n f(n,eps) * sum_{y: |I(x,y)|=n} phi(y)]

This is a linear operator (matrix) on functions on the causal set.
Its eigenvalues define a return probability and hence a spectral dimension.

We compare three operators:
1. Link-graph Laplacian (what everyone uses — gives wrong answer)
2. BD d'Alembertian (should give right answer)
3. SJ-derived operator (from the Pauli-Jordan function)
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from causal_sets.fast_core import sprinkle_fast, spectral_dimension_fast
from causal_sets.two_orders import TwoOrder
from causal_sets.bd_action import count_intervals_by_size


def bd_dalembertian_matrix(cs, eps=0.12):
    """
    Construct the BD d'Alembertian as an N×N matrix.

    B[x,y] encodes how element y contributes to (Box phi)(x).

    For element x, the d'Alembertian acts as:
      (B phi)(x) = alpha * [phi(x) - 2*eps * sum_n f(n,eps) * sum_{y in I_n(x)} phi(y)]

    where I_n(x) = {y : x < y and |interior of [x,y]| = n} (future intervals from x)
    PLUS  I_n(x) = {y : y < x and |interior of [y,x]| = n} (past intervals from x)

    The retarded part uses only future intervals (x < y).
    The full d'Alembertian uses both past and future.

    We construct the RETARDED d'Alembertian (only future) since that's what
    defines the causal propagator. Then symmetrize for spectral dimension.
    """
    N = cs.n
    order = cs.order
    order_int = order.astype(np.int32)
    interval_matrix = order_int @ order_int  # interval_matrix[i,j] = |interior of [i,j]|

    # Compute f(n, eps)
    max_n = min(N, 20)

    def f_eps(n, eps):
        if abs(1 - eps) < 1e-10:
            return 1.0 if n == 0 else 0.0
        r = (1 - eps) ** n
        return r * (1 - 2 * eps * n / (1 - eps) +
                     eps ** 2 * n * (n - 1) / (2 * (1 - eps) ** 2))

    f_values = np.array([f_eps(n, eps) for n in range(max_n + 1)])

    # Build the matrix B
    # B[x,y] = contribution of phi(y) to (Box phi)(x)
    B = np.zeros((N, N))

    # Diagonal: B[x,x] = alpha (the "phi(x)" term)
    # Off-diagonal: B[x,y] = -alpha * 2 * eps * f(n, eps) if y is in an n-interval from x

    # For each pair (x,y) with x < y (future direction):
    for x in range(N):
        for y in range(N):
            if x == y:
                continue
            if order[x, y]:  # x < y (y is in future of x)
                n = interval_matrix[x, y]
                if n <= max_n:
                    B[x, y] -= 2 * eps * f_values[n]
            elif order[y, x]:  # y < x (y is in past of x)
                n = interval_matrix[y, x]
                if n <= max_n:
                    B[x, y] -= 2 * eps * f_values[n]

    # Diagonal
    for x in range(N):
        B[x, x] = 1.0

    # Overall factor: 4*eps (but this just scales eigenvalues, not d_s shape)
    B *= 4 * eps

    return B


def spectral_dimension_from_operator(operator, sigma_range=(0.01, 100.0), n_sigma=50):
    """
    Compute spectral dimension from an arbitrary operator via heat kernel.

    For operator L with eigenvalues {lambda_k}:
    P(sigma) = (1/N) * Tr(exp(-sigma * L)) = (1/N) * sum_k exp(-sigma * lambda_k)
    d_s(sigma) = -2 * d(ln P) / d(ln sigma)
    """
    eigenvalues = np.linalg.eigvalsh(operator)

    # Shift so minimum eigenvalue is 0 (like a Laplacian)
    eigenvalues = eigenvalues - eigenvalues.min()

    N = len(eigenvalues)
    sigmas = np.logspace(np.log10(sigma_range[0]), np.log10(sigma_range[1]), n_sigma)

    P = np.zeros(n_sigma)
    for idx, sigma in enumerate(sigmas):
        P[idx] = np.mean(np.exp(-eigenvalues * sigma))

    ln_P = np.log(P + 1e-300)
    ln_sigma = np.log(sigmas)
    d_s = -2 * np.gradient(ln_P, ln_sigma)

    return sigmas, d_s


def main():
    rng = np.random.default_rng(42)

    print("=" * 75)
    print("EXPERIMENT 37: BD d'Alembertian Spectral Dimension")
    print("=" * 75)

    # Phase 1: Compare three operators on sprinkled 2D causets
    print("\n--- Phase 1: Three operators on sprinkled 2D (N=50) ---")
    print(f"  {'Method':>25} {'Peak d_s':>10} {'Correct?':>10}")
    print("-" * 50)

    N = 50
    n_trials = 10

    for eps in [0.12, 0.3, 0.5]:
        link_peaks = []
        bd_peaks = []

        for _ in range(n_trials):
            cs, _ = sprinkle_fast(N, dim=2, extent_t=1.0, region='diamond', rng=rng)

            # 1. Link-graph spectral dimension (the wrong way)
            sigmas_link, ds_link = spectral_dimension_fast(
                cs, sigma_range=(0.01, 100.0), n_sigma=50)
            if len(ds_link) > 0:
                link_peaks.append(np.max(ds_link))

            # 2. BD d'Alembertian spectral dimension
            B = bd_dalembertian_matrix(cs, eps=eps)
            # Symmetrize for spectral analysis
            B_sym = 0.5 * (B + B.T)
            sigmas_bd, ds_bd = spectral_dimension_from_operator(
                B_sym, sigma_range=(0.001, 50.0), n_sigma=60)
            if len(ds_bd) > 0:
                bd_peaks.append(np.max(ds_bd))

        print(f"  {'Link graph':>25} {np.mean(link_peaks):>10.2f} {'NO':>10}")
        label = f"BD dAlem (eps={eps})"
        correct = "YES" if abs(np.mean(bd_peaks) - 2.0) < 1.0 else "NO"
        print(f"  {label:>25} {np.mean(bd_peaks):>10.2f} {correct:>10}")

    # Phase 2: Scaling with N using best epsilon
    print(f"\n--- Phase 2: Scaling with N (eps=0.12) ---")
    print(f"  {'N':>5} {'d_s (link)':>12} {'d_s (BD)':>12}")
    print("-" * 35)

    for N in [20, 30, 50, 70, 100]:
        link_peaks = []
        bd_peaks = []

        for _ in range(8):
            cs, _ = sprinkle_fast(N, dim=2, extent_t=1.0, region='diamond', rng=rng)

            sigmas_l, ds_l = spectral_dimension_fast(cs, sigma_range=(0.01, 100.0), n_sigma=50)
            if len(ds_l) > 0:
                link_peaks.append(np.max(ds_l))

            B = bd_dalembertian_matrix(cs, eps=0.12)
            B_sym = 0.5 * (B + B.T)
            sigmas_b, ds_b = spectral_dimension_from_operator(
                B_sym, sigma_range=(0.001, 50.0), n_sigma=60)
            if len(ds_b) > 0:
                bd_peaks.append(np.max(ds_b))

        print(f"  {N:>5} {np.mean(link_peaks):>12.2f} {np.mean(bd_peaks):>12.2f}")

    # Phase 3: Full spectral dimension profile comparison
    print(f"\n--- Phase 3: Full d_s(sigma) profile (N=50, eps=0.12) ---")
    cs, _ = sprinkle_fast(50, dim=2, extent_t=1.0, region='diamond', rng=rng)

    sigmas_l, ds_l = spectral_dimension_fast(cs, sigma_range=(0.01, 100.0), n_sigma=50)
    B = bd_dalembertian_matrix(cs, eps=0.12)
    B_sym = 0.5 * (B + B.T)
    sigmas_b, ds_b = spectral_dimension_from_operator(B_sym, sigma_range=(0.001, 50.0), n_sigma=50)

    print(f"  {'sigma':>10} {'d_s (link)':>12} {'d_s (BD)':>12}")
    print("  " + "-" * 38)

    # Align sigma values for comparison
    for target_sigma in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        idx_l = np.argmin(np.abs(sigmas_l - target_sigma)) if len(sigmas_l) > 0 else -1
        idx_b = np.argmin(np.abs(sigmas_b - target_sigma)) if len(sigmas_b) > 0 else -1

        ds_l_val = ds_l[idx_l] if idx_l >= 0 else float('nan')
        ds_b_val = ds_b[idx_b] if idx_b >= 0 else float('nan')

        print(f"  {target_sigma:>10.3f} {ds_l_val:>12.2f} {ds_b_val:>12.2f}")

    # Phase 4: Does BD d'Alembertian d_s pass the random graph control?
    print(f"\n--- Phase 4: Random graph control ---")
    print("  If BD d'Alembertian d_s differs from random graphs: it captures geometry")

    for graph_type in ['Sprinkled 2D', 'Random DAG']:
        peaks = []
        for _ in range(10):
            if graph_type == 'Sprinkled 2D':
                cs, _ = sprinkle_fast(50, dim=2, extent_t=1.0, region='diamond', rng=rng)
            else:
                from causal_sets.fast_core import FastCausalSet
                adj = np.zeros((50, 50), dtype=bool)
                # Match link density of sprinkled 2D (~2.5 links/element)
                p = 2.5 * 2 / 49
                for i in range(50):
                    for j in range(i + 1, 50):
                        if rng.random() < p:
                            adj[i, j] = True
                cs = FastCausalSet(50)
                cs.order = adj

            B = bd_dalembertian_matrix(cs, eps=0.12)
            B_sym = 0.5 * (B + B.T)
            sigmas, ds = spectral_dimension_from_operator(
                B_sym, sigma_range=(0.001, 50.0), n_sigma=50)
            if len(ds) > 0:
                peaks.append(np.max(ds))

        print(f"  {graph_type:>15}: BD d_s peak = {np.mean(peaks):.2f} ± {np.std(peaks):.2f}")

    print("\n  If sprinkled ≈ 2 and random ≠ 2: BD d'Alembertian captures geometry!")
    print("  If both ≈ same: BD d'Alembertian has same problem as link graph")

    print("\n" + "=" * 75)
    print("DONE")
    print("=" * 75)


if __name__ == '__main__':
    main()

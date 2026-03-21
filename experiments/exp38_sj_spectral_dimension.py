"""
Experiment 38: Spectral Dimension from the SJ Vacuum

The link-graph Laplacian gives wrong d_s on causets (B1).
The BD d'Alembertian also fails (exp37).
Can the SJ Wightman function W define a spectral dimension that works?

Three approaches:
1. Use W eigenvalues directly: P(σ) = (1/N) Σ exp(-σ*ν_k)
2. Use (iΔ)² as a "causal Laplacian": L = Δ^T Δ (positive semi-definite)
3. Use -ln(W/(1-W)) as a "modular Hamiltonian" (from entanglement thermodynamics)

The key test: does the SJ spectral dimension give d_s ≈ 2 for sprinkled 2D causets?
If yes and random graphs give d_s ≠ 2, we've found a working probe.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from causal_sets.fast_core import sprinkle_fast, FastCausalSet
from causal_sets.two_orders import TwoOrder


def sj_operators(cs):
    """Compute SJ-derived operators for spectral dimension."""
    N = cs.n
    C = cs.order.astype(float)

    # Pauli-Jordan (antisymmetric)
    Delta = 0.5 * (C.T - C)

    # Approach 1: Causal Laplacian = Delta^T @ Delta (positive semi-definite)
    L_causal = Delta.T @ Delta

    # Approach 2: Wightman function eigenvalues
    H = 1j * Delta
    evals, evecs = np.linalg.eigh(H)
    evals = evals.real
    pos = evals > 1e-10
    W = (evecs[:, pos] @ np.diag(evals[pos]) @ evecs[:, pos].conj().T).real
    # Normalize
    w_max = np.linalg.eigvalsh(W).max()
    if w_max > 1:
        W = W / w_max

    return L_causal, W


def spectral_dim_from_operator(evals, sigma_range=(0.001, 100.0), n_sigma=60):
    """Compute d_s from eigenvalues of a positive semi-definite operator."""
    evals = evals[evals > 1e-12]  # remove zero modes
    if len(evals) == 0:
        return np.array([]), np.array([])

    # Shift minimum to 0
    evals = evals - evals.min()
    evals = evals[evals > 1e-12]
    if len(evals) == 0:
        return np.array([]), np.array([])

    N = len(evals)
    sigmas = np.logspace(np.log10(sigma_range[0]), np.log10(sigma_range[1]), n_sigma)
    P = np.zeros(n_sigma)
    for i, s in enumerate(sigmas):
        P[i] = np.mean(np.exp(-evals * s))

    ln_P = np.log(P + 1e-300)
    ln_s = np.log(sigmas)
    d_s = -2 * np.gradient(ln_P, ln_s)
    return sigmas, d_s


def main():
    rng = np.random.default_rng(42)

    print("=" * 75)
    print("EXPERIMENT 38: SJ Spectral Dimension")
    print("=" * 75)

    # Phase 1: Test all approaches on sprinkled 2D
    print("\n--- Phase 1: Sprinkled 2D causets (N=50) — target d_s ≈ 2 ---")
    N = 50
    n_trials = 15

    link_peaks = []
    causal_lap_peaks = []
    wightman_peaks = []

    for _ in range(n_trials):
        cs, _ = sprinkle_fast(N, dim=2, extent_t=1.0, region='diamond', rng=rng)
        L_causal, W = sj_operators(cs)

        # Link graph (the wrong way)
        from causal_sets.fast_core import spectral_dimension_fast
        s_l, ds_l = spectral_dimension_fast(cs, sigma_range=(0.01, 100.0), n_sigma=50)
        if len(ds_l) > 0:
            link_peaks.append(np.max(ds_l))

        # Causal Laplacian Δ^T Δ
        evals_cl = np.linalg.eigvalsh(L_causal)
        s_cl, ds_cl = spectral_dim_from_operator(evals_cl)
        if len(ds_cl) > 0:
            causal_lap_peaks.append(np.max(ds_cl))

        # Wightman eigenvalues
        evals_w = np.linalg.eigvalsh(W)
        s_w, ds_w = spectral_dim_from_operator(evals_w, sigma_range=(0.1, 1000.0))
        if len(ds_w) > 0:
            wightman_peaks.append(np.max(ds_w))

    print(f"  {'Method':>25} {'Peak d_s':>10} {'Correct?':>10}")
    print("-" * 50)
    print(f"  {'Link graph':>25} {np.mean(link_peaks):>10.2f} {'NO':>10}")
    print(f"  {'Causal Lap (Delta^T Delta)':>25} {np.mean(causal_lap_peaks):>10.2f} "
          f"{'YES' if abs(np.mean(causal_lap_peaks) - 2.0) < 0.5 else 'NO':>10}")
    print(f"  {'Wightman eigenvalues':>25} {np.mean(wightman_peaks):>10.2f} "
          f"{'YES' if abs(np.mean(wightman_peaks) - 2.0) < 0.5 else 'NO':>10}")

    # Phase 2: Random graph control
    print(f"\n--- Phase 2: Random graph control (N={N}) ---")

    for method_name, method_func in [
        ("Causal Laplacian", lambda cs: np.linalg.eigvalsh(sj_operators(cs)[0])),
        ("Wightman", lambda cs: np.linalg.eigvalsh(sj_operators(cs)[1])),
    ]:
        peaks_causet = []
        peaks_random = []

        for _ in range(15):
            # Sprinkled causet
            cs, _ = sprinkle_fast(N, dim=2, extent_t=1.0, region='diamond', rng=rng)
            evals = method_func(cs)
            s, ds = spectral_dim_from_operator(evals, sigma_range=(0.01, 1000.0))
            if len(ds) > 0:
                peaks_causet.append(np.max(ds))

            # Random DAG with matched link density
            links = int(np.sum(cs.link_matrix()))
            p = 2 * (links / N) / (N - 1)
            adj = np.zeros((N, N), dtype=bool)
            for i in range(N):
                for j in range(i + 1, N):
                    if rng.random() < p:
                        adj[i, j] = True
            rcs = FastCausalSet(N)
            rcs.order = adj
            evals_r = method_func(rcs)
            s_r, ds_r = spectral_dim_from_operator(evals_r, sigma_range=(0.01, 1000.0))
            if len(ds_r) > 0:
                peaks_random.append(np.max(ds_r))

        c_mean = np.mean(peaks_causet) if peaks_causet else float('nan')
        r_mean = np.mean(peaks_random) if peaks_random else float('nan')
        diff = abs(c_mean - r_mean) / max(abs(c_mean), 0.01)
        passes = "YES (different)" if diff > 0.15 else "NO (same)"

        print(f"  {method_name:>20}: causet={c_mean:.2f}, random={r_mean:.2f} "
              f"  Passes control? {passes}")

    # Phase 3: Scaling with N
    print(f"\n--- Phase 3: Scaling with N (best method) ---")
    print(f"  {'N':>5} {'Link graph':>12} {'Causal Lap':>12} {'Wightman':>12}")
    print("-" * 45)

    for N in [20, 30, 50, 70]:
        lp, clp, wp = [], [], []
        for _ in range(10):
            cs, _ = sprinkle_fast(N, dim=2, extent_t=1.0, region='diamond', rng=rng)
            L_causal, W = sj_operators(cs)

            s, ds = spectral_dimension_fast(cs, sigma_range=(0.01, 100.0), n_sigma=50)
            if len(ds) > 0: lp.append(np.max(ds))

            evals = np.linalg.eigvalsh(L_causal)
            s, ds = spectral_dim_from_operator(evals)
            if len(ds) > 0: clp.append(np.max(ds))

            evals = np.linalg.eigvalsh(W)
            s, ds = spectral_dim_from_operator(evals, sigma_range=(0.1, 1000.0))
            if len(ds) > 0: wp.append(np.max(ds))

        print(f"  {N:>5} {np.mean(lp):>12.2f} {np.mean(clp):>12.2f} {np.mean(wp):>12.2f}")

    print("\n  Target: constant ≈ 2 for a working estimator")
    print("  Link graph: diverges with N (BAD)")
    print("  Good estimator: stable near 2 as N grows")

    print("\n" + "=" * 75)
    print("DONE")
    print("=" * 75)


if __name__ == '__main__':
    main()

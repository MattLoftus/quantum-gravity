"""
Experiment 35: SJ Vacuum Entanglement Entropy at the BD Transition

Using the Sorkin-Johnston vacuum state for a free massless scalar field
on 2-order causal sets, compute the entanglement entropy of spatial
sub-regions across the BD phase transition.

Key questions:
1. Does S(A) follow an area law (S ~ boundary) or volume law (S ~ |A|)?
2. Does the entanglement structure differ between continuum and crystalline phases?
3. Is the continuum phase "holographic" (RT-like)?
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
import time
from causal_sets.two_orders import TwoOrder
from causal_sets.two_orders_v2 import mcmc_corrected
from causal_sets.sj_vacuum import sj_wightman_function, entanglement_entropy
from causal_sets.fast_core import sprinkle_fast


def spatial_partition_v(two_order, frac):
    """Partition by v-coordinate (spatial direction in a 2-order)."""
    v_order = np.argsort(two_order.v)
    cut = max(1, int(frac * two_order.N))
    A = list(v_order[:cut])
    return A


def main():
    rng = np.random.default_rng(42)
    N = 40  # Keep small for eigendecomposition speed
    eps = 0.12

    print("=" * 70)
    print("EXPERIMENT 35: SJ Vacuum Entanglement at the BD Transition")
    print(f"N={N}, eps={eps}")
    print("=" * 70)

    # Phase 1: Entanglement profile for sprinkled 2D causet (ground truth)
    print("\n--- Phase 1: Sprinkled 2D (ground truth) ---")
    print(f"  {'frac':>6} {'S(A)':>8} {'n_samples':>10}")
    print("-" * 30)

    for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        S_vals = []
        for _ in range(20):
            to = TwoOrder(N, rng=rng)
            cs = to.to_causet()
            W = sj_wightman_function(cs)
            A = spatial_partition_v(to, frac)
            S = entanglement_entropy(W, A)
            S_vals.append(S)
        print(f"  {frac:>6.1f} {np.mean(S_vals):>8.3f} ± {np.std(S_vals):>5.3f}")

    # Phase 2: Scaling with N (area vs volume law)
    print("\n--- Phase 2: S(N/2) vs N (area law: constant; volume law: linear) ---")
    print(f"  {'N':>5} {'S(N/2)':>8} {'S/N':>8} {'S/ln(N)':>8}")
    print("-" * 35)

    for N_test in [15, 20, 30, 40, 50]:
        S_vals = []
        for _ in range(15):
            to = TwoOrder(N_test, rng=rng)
            cs = to.to_causet()
            W = sj_wightman_function(cs)
            A = spatial_partition_v(to, 0.5)
            S = entanglement_entropy(W, A)
            S_vals.append(S)
        S_mean = np.mean(S_vals)
        print(f"  {N_test:>5} {S_mean:>8.3f} {S_mean/N_test:>8.4f} "
              f"{S_mean/np.log(N_test):>8.3f}")

    # Phase 3: Continuum vs Crystalline phase
    print("\n--- Phase 3: Continuum vs Crystalline ---")

    beta_c = 6.64 / (N * eps ** 2)

    for beta, label in [(0, "Continuum (β=0)"),
                         (beta_c * 2, f"Near transition (β={beta_c*2:.1f})"),
                         (beta_c * 5, f"Crystalline (β={beta_c*5:.1f})")]:
        t0 = time.time()

        res = mcmc_corrected(N, beta=beta, eps=eps,
                              n_steps=30000, n_therm=15000,
                              record_every=100, rng=rng)

        S_half_vals = []
        for cs in res['samples'][-20:]:
            W = sj_wightman_function(cs)
            A = list(range(N // 2))  # simple index partition
            S = entanglement_entropy(W, A)
            S_half_vals.append(S)

        elapsed = time.time() - t0
        print(f"\n  {label}:")
        print(f"    S(N/2) = {np.mean(S_half_vals):.3f} ± {np.std(S_half_vals):.3f}")
        print(f"    accept = {res['accept_rate']:.3f}, ({elapsed:.0f}s)")

    # Phase 4: Full entanglement profile in each phase
    print("\n--- Phase 4: Entanglement profiles ---")

    for beta, label in [(0, "Continuum"), (beta_c * 5, "Crystalline")]:
        res = mcmc_corrected(N, beta=beta, eps=eps,
                              n_steps=20000, n_therm=10000,
                              record_every=200, rng=rng)

        print(f"\n  {label} (β={beta:.1f}):")
        print(f"  {'frac':>6} {'S(A)':>8}")

        cs_sample = res['samples'][-1]
        W = sj_wightman_function(cs_sample)

        for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            k = max(1, int(frac * N))
            A = list(range(k))
            S = entanglement_entropy(W, A)
            print(f"  {frac:>6.1f} {S:>8.3f}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()

"""
Experiment 07: Scan the CSG coupling constant space for 4D manifold-like causets.

The transitive percolation model (t_k = t^k) produces 1D causets.
But the full CSG dynamics allows arbitrary coupling constants {t_n}.

Key question: do ANY coupling constants produce causets with:
- Myrheim-Meyer dimension ≈ 4
- Ordering fraction ≈ 0.10 (the theoretical value for 4D)
- Longest chain scaling as N^(1/4)

If yes: which coupling constants? What physics do they encode?
If no: CSG dynamics fundamentally cannot produce 4D universes.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from causal_sets.general_csg import scan_coupling_space, general_csg, _evaluate_couplings


def main():
    rng = np.random.default_rng(42)
    n = 200  # Start small for the scan (general CSG is expensive)
    n_trials = 3

    print("=" * 80)
    print("EXPERIMENT 07: CSG Coupling Constant Scan")
    print(f"N = {n}, trials = {n_trials}")
    print("=" * 80)

    print(f"\nTarget for 4D: ordering_fraction ≈ 0.10, MM_dim ≈ 4.0, chain ~ N^(1/4) ≈ {n**0.25:.1f}")
    print(f"Target for 3D: ordering_fraction ≈ 0.23, MM_dim ≈ 3.0, chain ~ N^(1/3) ≈ {n**(1/3):.1f}")
    print(f"Target for 2D: ordering_fraction ≈ 0.50, MM_dim ≈ 2.0, chain ~ N^(1/2) ≈ {n**0.5:.1f}")

    print(f"\n{'Label':>25} {'ord_frac':>10} {'MM_dim':>10} {'chain':>8} "
          f"{'chain/N':>8} {'links/N':>8}")
    print("-" * 80)

    results = scan_coupling_space(n, n_trials=n_trials, rng=rng)

    # Sort by how close the MM dimension is to 4.0
    results.sort(key=lambda r: abs(r['mm_dimension'] - 4.0))

    for r in results:
        # Highlight promising results
        marker = ""
        if 3.0 <= r['mm_dimension'] <= 5.0:
            marker = " ***"
        elif 2.0 <= r['mm_dimension'] <= 3.0:
            marker = " **"

        print(f"  {r['label']:>23} {r['ordering_fraction']:>10.4f} "
              f"{r['mm_dimension']:>8.2f}±{r['mm_std']:>4.2f} "
              f"{r['longest_chain']:>8.1f} {r['chain_ratio']:>8.4f} "
              f"{r['links_per_element']:>8.2f}{marker}")

    # Phase 2: Zoom in on the most promising region
    print("\n" + "=" * 80)
    print("Phase 2: Zooming in on promising coupling families")
    print("=" * 80)

    # Find the best result
    best = results[0]
    print(f"\nBest result so far: {best['label']} with MM_dim = {best['mm_dimension']:.2f}")

    # Try more fine-grained variations around the best family
    # Also try some theoretically motivated couplings

    print(f"\n{'Label':>30} {'ord_frac':>10} {'MM_dim':>10} {'chain':>8} "
          f"{'links/N':>8}")
    print("-" * 75)

    # Theoretically motivated: t_k = Gamma(k+1) * f(k)
    # The "natural" CSG measure weights by the number of natural labelings
    for c in [0.1, 0.3, 0.5, 0.7, 1.0]:
        t = [c ** k / max(1, np.math.factorial(min(k, 20))) for k in range(n)]
        r = _evaluate_couplings(n, t, n_trials, rng, f"factorial c={c}")
        marker = " ***" if 3.0 <= r['mm_dimension'] <= 5.0 else ""
        print(f"  {r['label']:>28} {r['ordering_fraction']:>10.4f} "
              f"{r['mm_dimension']:>8.2f}±{r['mm_std']:>4.2f} "
              f"{r['longest_chain']:>8.1f} {r['links_per_element']:>8.2f}{marker}")

    # Bell-shaped: peak at some k_peak
    for k_peak in [2, 4, 6, 8]:
        for width in [1.0, 2.0, 3.0]:
            t = [np.exp(-(k - k_peak) ** 2 / (2 * width ** 2)) for k in range(n)]
            r = _evaluate_couplings(n, t, n_trials, rng,
                                    f"bell peak={k_peak} w={width}")
            marker = " ***" if 3.0 <= r['mm_dimension'] <= 5.0 else ""
            print(f"  {r['label']:>28} {r['ordering_fraction']:>10.4f} "
                  f"{r['mm_dimension']:>8.2f}±{r['mm_std']:>4.2f} "
                  f"{r['longest_chain']:>8.1f} {r['links_per_element']:>8.2f}{marker}")

    # Oscillating: t_k = cos(omega * k)^2 * exp(-alpha*k)
    for omega in [0.5, 1.0, 2.0]:
        for alpha in [0.1, 0.5]:
            t = [np.cos(omega * k) ** 2 * np.exp(-alpha * k) for k in range(n)]
            r = _evaluate_couplings(n, t, n_trials, rng,
                                    f"osc w={omega} a={alpha}")
            marker = " ***" if 3.0 <= r['mm_dimension'] <= 5.0 else ""
            print(f"  {r['label']:>28} {r['ordering_fraction']:>10.4f} "
                  f"{r['mm_dimension']:>8.2f}±{r['mm_std']:>4.2f} "
                  f"{r['longest_chain']:>8.1f} {r['links_per_element']:>8.2f}{marker}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()

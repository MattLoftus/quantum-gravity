"""
Experiment 13: Systematic scan of CSG coupling constant parameter space
for manifold-like causets.

Key question: Does ANY CSG dynamics produce genuinely manifold-like causets?

A genuine d-dimensional manifold-like causet should have ALL of:
  - MM dimension ≈ d
  - Spectral dimension peak ≈ d (from link graph diffusion)
  - Links per element ≈ O(d) (each element has ~2d neighbors)
  - Ordering fraction ≈ f_0(d) (Myrheim-Meyer formula)
  - Longest chain ~ N^(1/d)

Previous results:
  - Transitive percolation (t_k = t^k) → 1D at all t
  - Step k_max=1 → MM dim ~4.3 but spectral dim ~1.15 (tree-like, NOT manifold)
  - MM dimension ALONE is insufficient — can be fooled by tree structures

Strategy:
  1. Define a composite manifold-likeness score
  2. Grid search over parametric families: A*exp(-B*k), A*k^C*exp(-B*k), etc.
  3. Gradient-free optimization (Nelder-Mead) to refine best candidates
  4. Verify best candidates at larger N with multiple trials

References:
  - Rideout & Sorkin, Phys. Rev. D 61, 024002 (2000)
  - Myrheim (1978), Meyer (1988) — dimension estimators
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from math import lgamma
from scipy.optimize import minimize, differential_evolution
from causal_sets.general_csg import general_csg, compute_denominator
from causal_sets.fast_core import FastCausalSet, spectral_dimension_fast, sprinkle_fast
from causal_sets.core import CausalSet
from causal_sets.dimension import myrheim_meyer, _ordering_fraction_theory
import time


# =============================================================================
# Manifold-likeness score
# =============================================================================

def ordering_fraction_target(d):
    """Theoretical ordering fraction f_0(d) for d-dimensional causal diamond."""
    return _ordering_fraction_theory(d)


def manifold_score(mm_dim, spectral_peak, links_per_elem, ordering_frac,
                   longest_chain, n, d_target=4.0):
    """
    Composite manifold-likeness score. LOWER is better (0 = perfect manifold).

    Components:
      1. |MM_dim - d_target|: Myrheim-Meyer dimension match
      2. |spectral_peak - d_target|: spectral dimension match
      3. |links_per_elem - expected|: link density match (2d expected for d-dim)
      4. |ordering_frac - f_0(d)|: ordering fraction match
      5. |chain_scaling - 1/d|: longest chain scaling match

    Weights chosen so each term contributes ~equally for a 1-unit error.
    """
    # Expected values for a d-dimensional manifold
    f0 = ordering_fraction_target(d_target)
    expected_links = 2 * d_target  # ~8 for 4D, ~6 for 3D, ~4 for 2D
    expected_chain_ratio = n ** (1.0 / d_target)

    # Component errors (normalized)
    err_mm = abs(mm_dim - d_target) / d_target  # fractional error
    err_spectral = abs(spectral_peak - d_target) / d_target
    err_links = abs(links_per_elem - expected_links) / expected_links
    err_ordering = abs(ordering_frac - 2 * f0) / (2 * f0)  # factor 2: our of convention
    err_chain = abs(longest_chain - expected_chain_ratio) / expected_chain_ratio

    # Weighted sum — spectral dimension is the hardest to fake
    score = (
        1.0 * err_mm +
        2.0 * err_spectral +  # double weight: tree structures fool MM but not spectral
        1.0 * err_links +
        0.5 * err_ordering +
        0.5 * err_chain
    )
    return score


def evaluate_coupling(t_vector, n, n_trials, rng, d_target=4.0, label=""):
    """
    Evaluate a coupling constant vector by growing causets and measuring
    all manifold-likeness observables.
    """
    mm_dims = []
    spectral_peaks = []
    links_per = []
    ord_fracs = []
    chains = []

    for trial in range(n_trials):
        cs = general_csg(n, list(t_vector), rng=rng)

        # Ordering fraction
        of = cs.ordering_fraction()
        ord_fracs.append(of)

        # MM dimension
        cs_old = CausalSet(n)
        cs_old.order = cs.order.astype(np.int8)
        d = myrheim_meyer(cs_old)
        mm_dims.append(d)

        # Links per element
        link_mat = cs.link_matrix()
        n_links = int(np.sum(link_mat))
        links_per.append(n_links / n)

        # Longest chain
        chains.append(cs.longest_chain())

        # Spectral dimension (on last trial only to save time, or all if n_trials small)
        if trial == n_trials - 1 or n_trials <= 2:
            sigmas, d_s = spectral_dimension_fast(cs, sigma_range=(0.01, 100.0), n_sigma=50)
            if len(d_s) > 5:
                # Peak spectral dimension (ignore edges which are noisy)
                core = d_s[3:-3]
                spectral_peaks.append(float(np.max(core)))
            else:
                spectral_peaks.append(0.0)

    result = {
        'label': label,
        't_vector': list(t_vector[:10]),  # truncate for display
        'mm_dim': np.mean(mm_dims),
        'mm_std': np.std(mm_dims),
        'spectral_peak': np.mean(spectral_peaks),
        'links_per_elem': np.mean(links_per),
        'ordering_frac': np.mean(ord_fracs),
        'longest_chain': np.mean(chains),
        'n': n,
    }

    result['score'] = manifold_score(
        result['mm_dim'], result['spectral_peak'], result['links_per_elem'],
        result['ordering_frac'], result['longest_chain'], n, d_target
    )

    return result


# =============================================================================
# Parametric coupling families
# =============================================================================

def make_t_exp(A, B, n):
    """t_k = A * exp(-B * k)"""
    return [A * np.exp(-B * k) for k in range(n)]


def make_t_exp_poly(A, B, C, n):
    """t_k = A * (k+1)^C * exp(-B * k)"""
    return [A * (k + 1) ** C * np.exp(-B * k) for k in range(n)]


def make_t_step_weighted(weights, n):
    """t_k = weights[k] for k < len(weights), 0 otherwise"""
    t = list(weights) + [0.0] * max(0, n - len(weights))
    return t


def make_t_bell(A, k_peak, width, n):
    """t_k = A * exp(-(k - k_peak)^2 / (2*width^2))"""
    return [A * np.exp(-(k - k_peak) ** 2 / (2 * width ** 2)) for k in range(n)]


def make_t_mixed(t0, t1, t2, decay, n):
    """First 3 couplings free, then exponential decay"""
    t = [t0, t1, t2] + [t2 * np.exp(-decay * (k - 2)) for k in range(3, n)]
    return t


# =============================================================================
# Phase 1: Coarse grid search
# =============================================================================

def phase1_grid_search(n=80, n_trials=2, d_target=4.0):
    """Coarse grid search over parametric families."""
    rng = np.random.default_rng(42)
    results = []

    print("=" * 90)
    print(f"PHASE 1: Coarse Grid Search (N={n}, {n_trials} trials, target d={d_target})")
    print("=" * 90)
    header = (f"  {'Label':>35} {'score':>7} {'MM':>6} {'spec':>6} "
              f"{'lnk/e':>6} {'ord_f':>7} {'chain':>6}")
    print(header)
    print("-" * 90)

    # Family 1: Exponential decay t_k = A * exp(-B*k)
    print("\n  --- Exponential: t_k = A * exp(-B*k) ---")
    for A in [0.01, 0.1, 1.0, 10.0, 100.0]:
        for B in [0.1, 0.5, 1.0, 2.0, 5.0]:
            t = make_t_exp(A, B, n)
            label = f"exp A={A} B={B}"
            r = evaluate_coupling(t, n, n_trials, rng, d_target, label)
            results.append(r)
            _print_result(r)

    # Family 2: Exp-polynomial t_k = A * (k+1)^C * exp(-B*k)
    print("\n  --- Exp-poly: t_k = A * (k+1)^C * exp(-B*k) ---")
    for A in [0.1, 1.0, 10.0]:
        for B in [0.3, 1.0, 3.0]:
            for C in [-2.0, -1.0, 0.0, 1.0, 2.0]:
                t = make_t_exp_poly(A, B, C, n)
                label = f"exppoly A={A} B={B} C={C}"
                r = evaluate_coupling(t, n, n_trials, rng, d_target, label)
                results.append(r)
                _print_result(r)

    # Family 3: Bell-shaped t_k peaked at k_peak
    print("\n  --- Bell: t_k = A * exp(-(k-k_peak)^2 / (2*w^2)) ---")
    for k_peak in [1, 2, 3, 5, 8]:
        for width in [0.5, 1.0, 2.0, 4.0]:
            for A in [0.1, 1.0, 10.0]:
                t = make_t_bell(A, k_peak, width, n)
                label = f"bell p={k_peak} w={width} A={A}"
                r = evaluate_coupling(t, n, n_trials, rng, d_target, label)
                results.append(r)
                _print_result(r)

    # Family 4: Step functions with varying heights
    print("\n  --- Step: first few couplings nonzero ---")
    for k_max in [1, 2, 3, 4, 5]:
        for val in [0.1, 0.5, 1.0, 5.0]:
            t = [val] * (k_max + 1) + [0.0] * (n - k_max - 1)
            label = f"step k={k_max} v={val}"
            r = evaluate_coupling(t, n, n_trials, rng, d_target, label)
            results.append(r)
            _print_result(r)

    # Family 5: Mixed first-3-free + decay
    print("\n  --- Mixed: t0,t1,t2 free, then exp decay ---")
    for t0 in [0.5, 1.0, 5.0]:
        for t1 in [0.5, 1.0, 5.0]:
            for t2 in [0.1, 1.0]:
                for decay in [0.5, 2.0]:
                    t = make_t_mixed(t0, t1, t2, decay, n)
                    label = f"mix {t0}/{t1}/{t2} d={decay}"
                    r = evaluate_coupling(t, n, n_trials, rng, d_target, label)
                    results.append(r)
                    _print_result(r)

    # Family 6: Power-law decay t_k = A / (k+1)^p
    print("\n  --- Power-law: t_k = A / (k+1)^p ---")
    for A in [0.1, 1.0, 10.0]:
        for p in [0.5, 1.0, 1.5, 2.0, 3.0]:
            t = [A / (k + 1) ** p for k in range(n)]
            label = f"power A={A} p={p}"
            r = evaluate_coupling(t, n, n_trials, rng, d_target, label)
            results.append(r)
            _print_result(r)

    return results


# =============================================================================
# Phase 2: Gradient-free optimization around best candidates
# =============================================================================

def phase2_optimize(best_results, n=80, n_trials=2, d_target=4.0):
    """Use Nelder-Mead and differential evolution to refine the best candidates."""
    rng = np.random.default_rng(123)

    print("\n" + "=" * 90)
    print(f"PHASE 2: Gradient-Free Optimization (N={n}, target d={d_target})")
    print("=" * 90)

    optimized = []

    # Strategy A: Optimize in the exponential family (A, B)
    print("\n--- Strategy A: Optimize exp(A, B) ---")

    def obj_exp(params):
        A, B = np.exp(params[0]), np.exp(params[1])  # log-space for positivity
        t = make_t_exp(A, B, n)
        r = evaluate_coupling(t, n, 1, rng, d_target, "opt_exp")
        return r['score']

    res = minimize(obj_exp, x0=[0.0, 0.0], method='Nelder-Mead',
                   options={'maxiter': 80, 'xatol': 0.1, 'fatol': 0.01})
    A_opt, B_opt = np.exp(res.x[0]), np.exp(res.x[1])
    t_opt = make_t_exp(A_opt, B_opt, n)
    r = evaluate_coupling(t_opt, n, n_trials, rng, d_target,
                          f"OPT exp A={A_opt:.3f} B={B_opt:.3f}")
    optimized.append(r)
    _print_result(r, prefix="  OPTIMIZED: ")

    # Strategy B: Optimize in exp-poly family (A, B, C)
    print("\n--- Strategy B: Optimize exp-poly(A, B, C) ---")

    def obj_exppoly(params):
        A, B = np.exp(params[0]), np.exp(params[1])
        C = params[2]
        t = make_t_exp_poly(A, B, C, n)
        r = evaluate_coupling(t, n, 1, rng, d_target, "opt_exppoly")
        return r['score']

    res = minimize(obj_exppoly, x0=[0.0, 0.0, 0.0], method='Nelder-Mead',
                   options={'maxiter': 120, 'xatol': 0.1, 'fatol': 0.01})
    A_opt, B_opt = np.exp(res.x[0]), np.exp(res.x[1])
    C_opt = res.x[2]
    t_opt = make_t_exp_poly(A_opt, B_opt, C_opt, n)
    r = evaluate_coupling(t_opt, n, n_trials, rng, d_target,
                          f"OPT exppoly A={A_opt:.3f} B={B_opt:.3f} C={C_opt:.3f}")
    optimized.append(r)
    _print_result(r, prefix="  OPTIMIZED: ")

    # Strategy C: Optimize free t_0..t_5 directly (6 free couplings)
    print("\n--- Strategy C: Optimize 6 free couplings ---")

    def obj_free6(params):
        # Softplus to ensure non-negative
        t_vals = [np.log(1 + np.exp(p)) for p in params]
        t = make_t_step_weighted(t_vals, n)
        r = evaluate_coupling(t, n, 1, rng, d_target, "opt_free6")
        return r['score']

    # Initialize from the best grid result
    best_grid = sorted(best_results, key=lambda r: r['score'])[:3]
    for i, bg in enumerate(best_grid):
        x0 = [np.log(np.exp(max(v, 0.001)) - 1) for v in bg['t_vector'][:6]]
        x0 = x0 + [0.0] * max(0, 6 - len(x0))

        res = minimize(obj_free6, x0=x0[:6], method='Nelder-Mead',
                       options={'maxiter': 150, 'xatol': 0.05, 'fatol': 0.01})

        t_vals = [np.log(1 + np.exp(p)) for p in res.x]
        t_opt = make_t_step_weighted(t_vals, n)
        r = evaluate_coupling(t_opt, n, n_trials, rng, d_target,
                              f"OPT free6 #{i+1}")
        optimized.append(r)
        _print_result(r, prefix="  OPTIMIZED: ")
        print(f"    t_vector: {[f'{v:.4f}' for v in t_vals]}")

    # Strategy D: Differential evolution on (A, B, C) with wider bounds
    print("\n--- Strategy D: Differential evolution on exp-poly ---")

    def obj_de(params):
        A, B, C = params
        t = make_t_exp_poly(A, B, C, n)
        r = evaluate_coupling(t, n, 1, rng, d_target, "DE")
        return r['score']

    try:
        res_de = differential_evolution(obj_de,
                                        bounds=[(0.001, 100), (0.01, 10), (-3, 3)],
                                        maxiter=30, seed=42, tol=0.01,
                                        popsize=8, mutation=(0.5, 1.5))
        A_de, B_de, C_de = res_de.x
        t_opt = make_t_exp_poly(A_de, B_de, C_de, n)
        r = evaluate_coupling(t_opt, n, n_trials, rng, d_target,
                              f"DE exppoly A={A_de:.3f} B={B_de:.3f} C={C_de:.3f}")
        optimized.append(r)
        _print_result(r, prefix="  DE BEST: ")
    except Exception as e:
        print(f"  DE failed: {e}")

    # Strategy E: Differential evolution on 6 free couplings
    print("\n--- Strategy E: Differential evolution on free t_0..t_5 ---")

    def obj_de_free(params):
        t = make_t_step_weighted(list(params), n)
        r = evaluate_coupling(t, n, 1, rng, d_target, "DE_free")
        return r['score']

    try:
        res_de = differential_evolution(obj_de_free,
                                        bounds=[(0.0, 20.0)] * 6,
                                        maxiter=30, seed=42, tol=0.01,
                                        popsize=8, mutation=(0.5, 1.5))
        t_vals = list(res_de.x)
        t_opt = make_t_step_weighted(t_vals, n)
        r = evaluate_coupling(t_opt, n, n_trials, rng, d_target,
                              f"DE free6")
        optimized.append(r)
        _print_result(r, prefix="  DE BEST: ")
        print(f"    t_vector: {[f'{v:.4f}' for v in t_vals]}")
    except Exception as e:
        print(f"  DE failed: {e}")

    return optimized


# =============================================================================
# Phase 3: Multi-target scan (also try d=2, d=3)
# =============================================================================

def phase2b_multi_target(n=80, n_trials=2):
    """Also scan for d=2 and d=3 manifold-like causets."""
    rng = np.random.default_rng(77)

    results_by_d = {}

    for d_target in [2.0, 3.0]:
        print(f"\n{'=' * 90}")
        print(f"MULTI-TARGET: Scanning for d={d_target:.0f} manifold-like causets (N={n})")
        f0 = ordering_fraction_target(d_target)
        expected_links = 2 * d_target
        print(f"  Targets: MM≈{d_target}, spec≈{d_target}, links/e≈{expected_links:.0f}, "
              f"ord_frac≈{2*f0:.4f}, chain≈{n**(1/d_target):.1f}")
        print("=" * 90)

        results = []

        # Quick scan of exponential family
        for A in [0.1, 1.0, 10.0]:
            for B in [0.1, 0.5, 1.0, 2.0, 5.0]:
                t = make_t_exp(A, B, n)
                r = evaluate_coupling(t, n, n_trials, rng, d_target,
                                      f"d{d_target:.0f} exp A={A} B={B}")
                results.append(r)
                if r['score'] < 2.0:
                    _print_result(r)

        # Exp-poly
        for A in [1.0, 10.0]:
            for B in [0.5, 1.0, 3.0]:
                for C in [-1.0, 0.0, 1.0]:
                    t = make_t_exp_poly(A, B, C, n)
                    r = evaluate_coupling(t, n, n_trials, rng, d_target,
                                          f"d{d_target:.0f} ep A={A} B={B} C={C}")
                    results.append(r)
                    if r['score'] < 2.0:
                        _print_result(r)

        # Optimize best
        best3 = sorted(results, key=lambda r: r['score'])[:3]
        print(f"\n  Top 3 for d={d_target:.0f}:")
        for r in best3:
            _print_result(r, prefix="    ")

        results_by_d[d_target] = results

    return results_by_d


# =============================================================================
# Phase 4: Verification of best candidates
# =============================================================================

def phase4_verify(candidates, d_target=4.0):
    """Verify best candidates at larger N with more trials and full diagnostics."""
    rng = np.random.default_rng(999)

    print("\n" + "=" * 90)
    print("PHASE 4: Verification of Best Candidates")
    print("=" * 90)

    for i, cand in enumerate(candidates):
        t_vec = cand['t_vector']
        label = cand['label']

        print(f"\n--- Candidate {i+1}: {label} ---")
        print(f"  t_vector (first 10): {[f'{v:.4f}' for v in t_vec[:10]]}")

        # Test at multiple sizes
        print(f"\n  {'N':>5} {'MM_dim':>8} {'spec_pk':>8} {'lnk/e':>7} "
              f"{'ord_f':>7} {'chain':>6} {'score':>7}")
        print("  " + "-" * 55)

        for n_test in [100, 200, 300, 500]:
            n_trials_v = 5 if n_test <= 300 else 3
            r = evaluate_coupling(t_vec, n_test, n_trials_v, rng, d_target,
                                  f"{label} N={n_test}")
            print(f"  {n_test:>5} {r['mm_dim']:>7.2f}±{r['mm_std']:.1f} "
                  f"{r['spectral_peak']:>8.2f} {r['links_per_elem']:>7.2f} "
                  f"{r['ordering_frac']:>7.4f} {r['longest_chain']:>6.1f} "
                  f"{r['score']:>7.3f}")

        # Full spectral dimension profile at N=300
        print(f"\n  Spectral dimension profile (N=300):")
        cs = general_csg(300, list(t_vec), rng=rng)
        sigmas, d_s = spectral_dimension_fast(cs, sigma_range=(0.005, 200.0), n_sigma=80)
        if len(d_s) > 0:
            peak_idx = np.argmax(d_s[3:-3]) + 3
            print(f"    Peak d_s = {d_s[peak_idx]:.3f} at sigma = {sigmas[peak_idx]:.4f}")
            print(f"    {'sigma':>10} {'d_s':>8}")
            for idx in np.linspace(3, len(sigmas) - 3, 12, dtype=int):
                marker = ""
                if abs(d_s[idx] - d_target) < 0.3:
                    marker = " <-- TARGET"
                elif abs(d_s[idx] - 2.0) < 0.3:
                    marker = " (d_s≈2)"
                print(f"    {sigmas[idx]:>10.4f} {d_s[idx]:>8.3f}{marker}")

        # Interval size distribution
        print(f"\n  Interval size distribution (N=300):")
        _, sizes = cs.interval_sizes_vectorized()
        if len(sizes) > 0:
            print(f"    n_intervals={len(sizes)}, mean={np.mean(sizes):.2f}, "
                  f"median={np.median(sizes):.0f}, max={np.max(sizes)}")
            # Histogram of small intervals
            hist, _ = np.histogram(sizes, bins=range(0, min(20, int(np.max(sizes)) + 2)))
            print(f"    Size distribution (0-19): {list(hist)}")

        # Compare with sprinkled reference
        print(f"\n  Comparison with sprinkled {d_target:.0f}D causet (N=300):")
        cs_ref, _ = sprinkle_fast(300, dim=int(d_target), extent_t=1.0,
                                  region='diamond', rng=rng)
        _, sizes_ref = cs_ref.interval_sizes_vectorized()
        sigmas_ref, ds_ref = spectral_dimension_fast(cs_ref, sigma_range=(0.005, 200.0),
                                                     n_sigma=80)

        cs_ref_old = CausalSet(300)
        cs_ref_old.order = cs_ref.order.astype(np.int8)

        print(f"    {'':>20} {'Sprinkled':>12} {'CSG':>12}")
        print(f"    {'Ordering frac':>20} {cs_ref.ordering_fraction():>12.4f} "
              f"{cs.ordering_fraction():>12.4f}")
        print(f"    {'MM dimension':>20} {myrheim_meyer(cs_ref_old):>12.2f} "
              f"{r['mm_dim']:>12.2f}")
        print(f"    {'Links/element':>20} "
              f"{int(np.sum(cs_ref.link_matrix()))/300:>12.2f} "
              f"{int(np.sum(cs.link_matrix()))/300:>12.2f}")
        print(f"    {'Longest chain':>20} {cs_ref.longest_chain():>12d} "
              f"{cs.longest_chain():>12d}")
        if len(sizes_ref) > 0:
            print(f"    {'Mean interval':>20} {np.mean(sizes_ref):>12.2f} "
                  f"{np.mean(sizes) if len(sizes) > 0 else 0:>12.2f}")
        if len(ds_ref) > 5:
            pk_ref = np.max(ds_ref[3:-3])
            print(f"    {'Spectral peak':>20} {pk_ref:>12.2f} "
                  f"{d_s[peak_idx] if len(d_s) > 5 else 0:>12.2f}")


# =============================================================================
# Helpers
# =============================================================================

def _print_result(r, prefix="  "):
    """Print a single result line."""
    marker = ""
    if r['score'] < 1.0:
        marker = " *** GOOD"
    elif r['score'] < 1.5:
        marker = " ** decent"
    elif r['score'] < 2.0:
        marker = " * ok"
    print(f"{prefix}{r['label']:>35} {r['score']:>7.3f} "
          f"{r['mm_dim']:>5.2f} {r['spectral_peak']:>5.2f} "
          f"{r['links_per_elem']:>5.2f} {r['ordering_frac']:>7.4f} "
          f"{r['longest_chain']:>5.1f}{marker}")


# =============================================================================
# Main
# =============================================================================

def main():
    t_start = time.time()

    print("=" * 90)
    print("EXPERIMENT 13: Systematic Scan of CSG Coupling Constant Space")
    print("=" * 90)
    print(f"\nKey question: Does ANY CSG dynamics produce manifold-like causets?")
    print(f"\nManifold-likeness requires ALL of:")
    print(f"  - MM dimension ≈ d")
    print(f"  - Spectral dimension peak ≈ d")
    print(f"  - Links per element ≈ 2d")
    print(f"  - Ordering fraction consistent with d-dim Myrheim-Meyer")
    print(f"  - Longest chain ~ N^(1/d)")

    for d in [2, 3, 4]:
        f0 = ordering_fraction_target(d)
        print(f"\n  d={d}: f_0={f0:.4f}, ord_frac≈{2*f0:.4f}, "
              f"links/e≈{2*d}, chain(N=80)≈{80**(1/d):.1f}")

    # ---- Phase 1: Grid search for d=4 ----
    results_grid = phase1_grid_search(n=80, n_trials=2, d_target=4.0)

    # Sort and display top results
    results_grid.sort(key=lambda r: r['score'])
    print(f"\n{'=' * 90}")
    print("TOP 20 CANDIDATES FROM GRID SEARCH (d=4 target)")
    print("=" * 90)
    header = (f"  {'Label':>35} {'score':>7} {'MM':>6} {'spec':>6} "
              f"{'lnk/e':>6} {'ord_f':>7} {'chain':>6}")
    print(header)
    print("-" * 90)
    for r in results_grid[:20]:
        _print_result(r)

    elapsed = time.time() - t_start
    print(f"\n  [Phase 1 took {elapsed:.0f}s]")

    # ---- Phase 2: Optimization ----
    optimized = phase2_optimize(results_grid[:10], n=80, n_trials=2, d_target=4.0)

    # ---- Phase 2b: Also try d=2, d=3 ----
    multi_results = phase2b_multi_target(n=80, n_trials=2)

    # ---- Collect all best candidates ----
    all_results = results_grid + optimized
    for d_key, res_list in multi_results.items():
        all_results.extend(res_list)

    # Separate best for each target dimension
    best_d4 = sorted([r for r in all_results if 'score' in r],
                     key=lambda r: r['score'])[:5]

    # ---- Phase 4: Verify top candidates ----
    print(f"\n{'=' * 90}")
    print("VERIFYING TOP 3 CANDIDATES")
    print("=" * 90)
    phase4_verify(best_d4[:3], d_target=4.0)

    # ---- Final summary ----
    elapsed = time.time() - t_start
    print(f"\n{'=' * 90}")
    print("FINAL SUMMARY")
    print(f"{'=' * 90}")
    print(f"Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Total configurations tested: {len(all_results)}")

    best = best_d4[0] if best_d4 else None
    if best:
        print(f"\nBest manifold-likeness score: {best['score']:.3f}")
        print(f"  Label: {best['label']}")
        print(f"  MM dimension: {best['mm_dim']:.2f}")
        print(f"  Spectral peak: {best['spectral_peak']:.2f}")
        print(f"  Links/element: {best['links_per_elem']:.2f}")
        print(f"  Ordering frac: {best['ordering_frac']:.4f}")
        print(f"  t_vector: {best['t_vector']}")

    # Interpretation
    print(f"\n{'=' * 90}")
    print("INTERPRETATION")
    print("=" * 90)
    if best and best['score'] < 1.0:
        print("  POSITIVE RESULT: Found CSG coupling constants producing")
        print(f"  manifold-like causets with score {best['score']:.3f}.")
        print("  This suggests CSG dynamics CAN produce spacetime-like structures.")
    elif best and best['score'] < 2.0:
        print("  MARGINAL RESULT: Best candidates show SOME manifold-like features")
        print(f"  (score {best['score']:.3f}) but are not convincingly manifold-like.")
        print("  CSG may need additional ingredients (e.g., non-locality, quantum")
        print("  measure theory, or modified dynamics) to produce realistic spacetimes.")
    else:
        print("  NEGATIVE RESULT: No CSG coupling constants found that produce")
        print("  manifold-like causets. This is an important result:")
        print("  CSG dynamics alone (with any coupling constants) appears insufficient")
        print("  to generate spacetime-like causal sets.")
        print("  ")
        print("  The core issue is likely that CSG is inherently 'sequential' —")
        print("  each element is added one at a time, creating tree-like or chain-like")
        print("  structures rather than the multi-dimensional connectivity needed")
        print("  for manifold-like geometry.")
        print("  ")
        print("  Possible extensions to explore:")
        print("    1. Quantum CSG (complex amplitudes, path integral over growths)")
        print("    2. Non-local CSG (couplings depend on global causet geometry)")
        print("    3. Post-selection on geometric constraints")
        print("    4. Hybrid models (CSG + spatial structure)")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == '__main__':
    main()

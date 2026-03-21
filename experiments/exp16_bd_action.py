"""
Experiment 16: Benincasa-Dowker action and MCMC sampling of causets.

Key question: Can MCMC with the BD action produce manifold-like causets?
CSG dynamics (exp13) are inherently tree-like — no coupling configuration
produces manifold-like causets. The BD action is a causal set discretization
of the Einstein-Hilbert action and should favor manifold-like configurations.

Plan:
  Part 1: Validate BD action
    - Compute S_BD for 2D sprinkled causets (should be ~0 for flat space)
    - Compute S_BD for random DAGs (should be large)
    - Compute S_BD for CSG causets (should be intermediate)

  Part 2: MCMC sampling
    - Start from sprinkled causet (known manifold)
    - Run MCMC at several beta values
    - Measure: action, MM dimension, spectral dimension, link density

  Part 3: Analysis
    - Does large beta preserve/restore manifold-like properties?
    - Compare MCMC samples vs sprinkled ground truth vs CSG

References:
  - Benincasa & Dowker, Phys. Rev. Lett. 104, 181301 (2010)
  - Glaser & Surya, Phys. Rev. D 94, 064014 (2016)
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
import time
from causal_sets.fast_core import (FastCausalSet, sprinkle_fast, csg_fast,
                                    spectral_dimension_fast)
from causal_sets.bd_action import bd_action_2d, count_intervals_by_size
from causal_sets.mcmc import mcmc_bd_action
from causal_sets.dimension import myrheim_meyer, _ordering_fraction_theory, _invert_ordering_fraction
from causal_sets.core import CausalSet


def fast_to_core(fc: FastCausalSet) -> CausalSet:
    """Convert FastCausalSet to CausalSet for dimension estimators."""
    c = CausalSet(fc.n)
    c.order = fc.order.astype(np.int8)
    return c


def measure_observables(causet: FastCausalSet) -> dict:
    """Measure key observables of a causet."""
    n = causet.n
    if n < 5:
        return {'n': n, 'mm_dim': float('nan'), 'spectral_peak': float('nan'),
                'links_per_elem': float('nan'), 'ordering_frac': float('nan'),
                'action': float('nan')}

    # MM dimension
    core_causet = fast_to_core(causet)
    mm_dim = myrheim_meyer(core_causet)

    # Links
    links = causet.link_matrix()
    n_links = int(np.sum(links))
    links_per_elem = n_links / n if n > 0 else 0

    # Ordering fraction
    of = causet.ordering_fraction()

    # Action
    action = bd_action_2d(causet)

    # Spectral dimension (peak)
    try:
        sigmas, ds = spectral_dimension_fast(causet, sigma_range=(0.1, 50.0), n_sigma=40)
        if len(ds) > 5:
            # Find peak in the middle range (avoid boundary effects)
            mid = len(ds) // 4
            end = 3 * len(ds) // 4
            spectral_peak = float(np.max(ds[mid:end]))
        else:
            spectral_peak = float('nan')
    except Exception:
        spectral_peak = float('nan')

    return {
        'n': n,
        'mm_dim': mm_dim,
        'spectral_peak': spectral_peak,
        'links_per_elem': links_per_elem,
        'ordering_frac': of,
        'action': action,
        'action_per_elem': action / n if n > 0 else float('nan'),
    }


def generate_random_dag(n: int, edge_prob: float = 0.3,
                        rng: np.random.Generator = None) -> FastCausalSet:
    """Generate a random DAG (not from any manifold)."""
    if rng is None:
        rng = np.random.default_rng()

    causet = FastCausalSet(n)
    # Random upper-triangular: each pair (i,j) with i<j gets a relation with prob edge_prob
    for i in range(n):
        rand = rng.random(n - i - 1)
        causet.order[i, i+1:] = rand < edge_prob

    # Transitive closure
    order_int = causet.order.astype(np.int32)
    for k in range(n):
        for i in range(k):
            if causet.order[i, k]:
                causet.order[i, k+1:] |= causet.order[k, k+1:]

    return causet


# =============================================================================
# Part 1: Validate BD action on known causets
# =============================================================================
def part1_validate():
    print("=" * 70)
    print("PART 1: Validate BD action on known causets")
    print("=" * 70)

    rng = np.random.default_rng(42)
    sizes = [30, 50, 80]
    n_trials = 5

    print("\n--- 2D Sprinkled causets (expected: S ~ 0 for flat 2D Minkowski) ---")
    for n in sizes:
        actions = []
        for trial in range(n_trials):
            causet, coords = sprinkle_fast(n, dim=2, rng=rng)
            s = bd_action_2d(causet)
            actions.append(s)
        actions = np.array(actions)
        print(f"  N={n:3d}: S_BD = {actions.mean():+7.2f} ± {actions.std():.2f}  "
              f"(S/N = {actions.mean()/n:+.4f})")

    print("\n--- Random DAGs (expected: |S| >> 0, non-manifold) ---")
    for n in sizes:
        for edge_prob in [0.1, 0.3, 0.5]:
            actions = []
            for trial in range(n_trials):
                causet = generate_random_dag(n, edge_prob=edge_prob, rng=rng)
                s = bd_action_2d(causet)
                actions.append(s)
            actions = np.array(actions)
            print(f"  N={n:3d}, p={edge_prob:.1f}: S_BD = {actions.mean():+7.2f} ± {actions.std():.2f}  "
                  f"(S/N = {actions.mean()/n:+.4f})")

    print("\n--- CSG causets (expected: intermediate, tree-like) ---")
    for n in sizes:
        for coupling in [0.1, 0.3, 0.5, 0.7]:
            actions = []
            for trial in range(n_trials):
                causet = csg_fast(n, coupling=coupling, rng=rng)
                s = bd_action_2d(causet)
                actions.append(s)
            actions = np.array(actions)
            print(f"  N={n:3d}, t={coupling:.1f}: S_BD = {actions.mean():+7.2f} ± {actions.std():.2f}  "
                  f"(S/N = {actions.mean()/n:+.4f})")

    print("\n--- Detailed observables for N=50 ---")
    print(f"  {'Type':<25s} {'S_BD':>7s} {'S/N':>7s} {'MM dim':>7s} {'d_s':>7s} {'L/N':>7s} {'f':>7s}")
    print("  " + "-" * 68)

    # Sprinkled 2D
    causet, _ = sprinkle_fast(50, dim=2, rng=rng)
    obs = measure_observables(causet)
    print(f"  {'Sprinkle 2D':<25s} {obs['action']:+7.2f} {obs['action_per_elem']:+7.4f} "
          f"{obs['mm_dim']:7.2f} {obs['spectral_peak']:7.2f} "
          f"{obs['links_per_elem']:7.2f} {obs['ordering_frac']:7.4f}")

    # Random DAG
    causet = generate_random_dag(50, edge_prob=0.3, rng=rng)
    obs = measure_observables(causet)
    print(f"  {'Random DAG p=0.3':<25s} {obs['action']:+7.2f} {obs['action_per_elem']:+7.4f} "
          f"{obs['mm_dim']:7.2f} {obs['spectral_peak']:7.2f} "
          f"{obs['links_per_elem']:7.2f} {obs['ordering_frac']:7.4f}")

    # CSG
    causet = csg_fast(50, coupling=0.3, rng=rng)
    obs = measure_observables(causet)
    print(f"  {'CSG t=0.3':<25s} {obs['action']:+7.2f} {obs['action_per_elem']:+7.4f} "
          f"{obs['mm_dim']:7.2f} {obs['spectral_peak']:7.2f} "
          f"{obs['links_per_elem']:7.2f} {obs['ordering_frac']:7.4f}")

    causet = csg_fast(50, coupling=0.5, rng=rng)
    obs = measure_observables(causet)
    print(f"  {'CSG t=0.5':<25s} {obs['action']:+7.2f} {obs['action_per_elem']:+7.4f} "
          f"{obs['mm_dim']:7.2f} {obs['spectral_peak']:7.2f} "
          f"{obs['links_per_elem']:7.2f} {obs['ordering_frac']:7.4f}")


# =============================================================================
# Part 2: MCMC sampling
# =============================================================================
def part2_mcmc():
    print("\n" + "=" * 70)
    print("PART 2: MCMC sampling with BD action")
    print("=" * 70)

    rng = np.random.default_rng(123)
    target_n = 50
    n_steps = 2000
    burn_in = 500

    # Start from a sprinkled causet
    initial, coords = sprinkle_fast(target_n, dim=2, rng=rng)
    initial_obs = measure_observables(initial)
    print(f"\nInitial (sprinkled 2D, N={target_n}):")
    print(f"  S_BD = {initial_obs['action']:.2f}, MM dim = {initial_obs['mm_dim']:.2f}, "
          f"d_s = {initial_obs['spectral_peak']:.2f}, L/N = {initial_obs['links_per_elem']:.2f}")

    beta_values = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

    print(f"\nRunning MCMC for {n_steps} steps at each beta (target N={target_n})...")
    print(f"  {'beta':>6s} {'accept':>7s} {'<S>':>8s} {'<S/N>':>8s} {'<N>':>6s} "
          f"{'<MM>':>7s} {'<d_s>':>7s} {'<L/N>':>7s}")
    print("  " + "-" * 62)

    results = {}

    for beta in beta_values:
        t0 = time.time()

        mcmc_result = mcmc_bd_action(
            initial_causet=initial,
            beta=beta,
            n_steps=n_steps,
            target_n=target_n,
            n_size_penalty=1.0,  # gentle penalty to keep size near target
            rng=rng,
            record_every=10,
            verbose=False,
        )

        # Measure observables on post-burn-in samples
        burn_idx = burn_in // 10
        post_burn_samples = mcmc_result['samples'][burn_idx:]
        post_burn_actions = mcmc_result['actions'][burn_idx:]
        post_burn_sizes = mcmc_result['sizes'][burn_idx:]

        # Measure detailed observables on a subsample
        n_measure = min(20, len(post_burn_samples))
        step_measure = max(1, len(post_burn_samples) // n_measure)
        mm_dims = []
        spectral_peaks = []
        links_per_elems = []

        for idx in range(0, len(post_burn_samples), step_measure):
            sample = post_burn_samples[idx]
            obs = measure_observables(sample)
            mm_dims.append(obs['mm_dim'])
            spectral_peaks.append(obs['spectral_peak'])
            links_per_elems.append(obs['links_per_elem'])

        mm_dims = np.array(mm_dims)
        spectral_peaks = np.array(spectral_peaks)
        links_per_elems = np.array(links_per_elems)

        # Filter NaN for means
        mm_mean = np.nanmean(mm_dims)
        sp_mean = np.nanmean(spectral_peaks)
        lpe_mean = np.nanmean(links_per_elems)

        elapsed = time.time() - t0

        print(f"  {beta:6.1f} {mcmc_result['accept_rate']:7.3f} "
              f"{post_burn_actions.mean():+8.2f} {(post_burn_actions / post_burn_sizes).mean():+8.4f} "
              f"{post_burn_sizes.mean():6.1f} "
              f"{mm_mean:7.2f} {sp_mean:7.2f} {lpe_mean:7.2f}  ({elapsed:.1f}s)")

        results[beta] = {
            'accept_rate': mcmc_result['accept_rate'],
            'mean_action': float(post_burn_actions.mean()),
            'mean_size': float(post_burn_sizes.mean()),
            'mean_mm_dim': float(mm_mean),
            'mean_spectral': float(sp_mean),
            'mean_links_per_elem': float(lpe_mean),
            'mm_dims': mm_dims,
            'spectral_peaks': spectral_peaks,
        }

    return results


# =============================================================================
# Part 3: Analysis — comparison with ground truth
# =============================================================================
def part3_analysis(mcmc_results: dict):
    print("\n" + "=" * 70)
    print("PART 3: Analysis — MCMC vs sprinkled ground truth")
    print("=" * 70)

    rng = np.random.default_rng(999)

    # Ground truth: 2D sprinkled causets
    print("\n--- Ground truth: 2D sprinkled causets (N=50, 20 samples) ---")
    gt_obs = []
    for _ in range(20):
        causet, _ = sprinkle_fast(50, dim=2, rng=rng)
        obs = measure_observables(causet)
        gt_obs.append(obs)

    gt_mm = np.array([o['mm_dim'] for o in gt_obs])
    gt_sp = np.array([o['spectral_peak'] for o in gt_obs])
    gt_lpe = np.array([o['links_per_elem'] for o in gt_obs])
    gt_action = np.array([o['action'] for o in gt_obs])

    print(f"  MM dimension:     {np.nanmean(gt_mm):.2f} ± {np.nanstd(gt_mm):.2f}")
    print(f"  Spectral dim:     {np.nanmean(gt_sp):.2f} ± {np.nanstd(gt_sp):.2f}")
    print(f"  Links/element:    {np.nanmean(gt_lpe):.2f} ± {np.nanstd(gt_lpe):.2f}")
    print(f"  BD action:        {np.nanmean(gt_action):.2f} ± {np.nanstd(gt_action):.2f}")
    print(f"  Action/element:   {np.nanmean(gt_action)/50:.4f} ± {np.nanstd(gt_action)/50:.4f}")

    # Manifold-likeness score for each beta
    print("\n--- Manifold-likeness comparison ---")
    print(f"  {'Source':<20s} {'MM dim':>8s} {'d_s':>8s} {'L/N':>8s} {'Score':>8s}")
    print("  " + "-" * 56)

    # Score = sum of squared deviations from 2D ground truth
    gt_targets = {
        'mm': np.nanmean(gt_mm),
        'sp': np.nanmean(gt_sp),
        'lpe': np.nanmean(gt_lpe),
    }

    def manifold_score(mm, sp, lpe):
        score = 0
        score += (mm - gt_targets['mm']) ** 2
        score += (sp - gt_targets['sp']) ** 2
        score += (lpe - gt_targets['lpe']) ** 2
        return score

    gt_score = manifold_score(gt_targets['mm'], gt_targets['sp'], gt_targets['lpe'])
    print(f"  {'Ground truth (2D)':<20s} {gt_targets['mm']:8.2f} {gt_targets['sp']:8.2f} "
          f"{gt_targets['lpe']:8.2f} {gt_score:8.4f}")

    for beta, res in sorted(mcmc_results.items()):
        score = manifold_score(res['mean_mm_dim'], res['mean_spectral'],
                               res['mean_links_per_elem'])
        print(f"  {'MCMC β='+str(beta):<20s} {res['mean_mm_dim']:8.2f} {res['mean_spectral']:8.2f} "
              f"{res['mean_links_per_elem']:8.2f} {score:8.4f}")

    # Key question
    print("\n--- KEY QUESTION: Does BD action MCMC produce manifold-like causets? ---")
    best_beta = min(mcmc_results.keys(),
                    key=lambda b: manifold_score(mcmc_results[b]['mean_mm_dim'],
                                                  mcmc_results[b]['mean_spectral'],
                                                  mcmc_results[b]['mean_links_per_elem']))
    best = mcmc_results[best_beta]
    print(f"  Best beta: {best_beta}")
    print(f"  Best MM dim: {best['mean_mm_dim']:.2f} (target: {gt_targets['mm']:.2f})")
    print(f"  Best d_s: {best['mean_spectral']:.2f} (target: {gt_targets['sp']:.2f})")
    print(f"  Best L/N: {best['mean_links_per_elem']:.2f} (target: {gt_targets['lpe']:.2f})")

    # Check if any beta achieves manifold-like properties
    mm_ok = abs(best['mean_mm_dim'] - gt_targets['mm']) < 0.5
    sp_ok = abs(best['mean_spectral'] - gt_targets['sp']) < 0.5
    lpe_ok = abs(best['mean_links_per_elem'] - gt_targets['lpe']) < 1.0

    if mm_ok and sp_ok and lpe_ok:
        print("\n  *** YES — MCMC with BD action produces manifold-like causets! ***")
        print("  This is a significant improvement over CSG dynamics.")
    elif mm_ok or sp_ok:
        print("\n  PARTIAL — Some manifold-like properties emerge, but not all.")
        print("  The BD action helps but may need longer chains or larger N.")
    else:
        print("\n  NOT YET — MCMC samples are not manifold-like at these parameters.")
        print("  May need: larger N, more MCMC steps, different move set, or")
        print("  different beta range.")


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    print("Experiment 16: Benincasa-Dowker Action & MCMC Causet Sampling")
    print("=" * 70)

    t_start = time.time()

    part1_validate()
    mcmc_results = part2_mcmc()
    part3_analysis(mcmc_results)

    t_total = time.time() - t_start
    print(f"\nTotal runtime: {t_total:.1f}s")

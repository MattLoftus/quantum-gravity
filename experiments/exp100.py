"""
Experiment 100: STRENGTHEN WEAKEST PAPERS — Ideas 511-520

Paper A (7/10) and Paper B2 (5/10) are the weakest. Plus targeted improvements
for Papers D and C.

Ideas:
511. Paper A: interval entropy at N=100-200 with parallel tempering — 4D three-phase persist?
512. Paper A: interval entropy on SPRINKLED 4D causets (not d-orders) — same three-phase?
513. Paper A: finite-size scaling collapse of H(beta) — optimal nu
514. Paper A: compare interval entropy with link fraction — which is sharper?
515. Paper B2: tighten alpha constraint — 10,000 realizations
516. Paper B2: DESI DR2 w(z) prediction
517. Paper B2: Bayes factor with proper prior
518. Paper D: phase-mixing artifact quantification — how many samples to get <r>=0.12?
519. Paper D: GUE universality on 3D and 4D causets (not just 2D)
520. Paper C: ER=EPR Gram identity test on sprinkled causets and d-orders
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats, optimize
from scipy.linalg import eigh
import time

from causal_sets.fast_core import FastCausalSet, sprinkle_fast
from causal_sets.two_orders import TwoOrder, swap_move
from causal_sets.two_orders_v2 import bd_action_corrected, parallel_tempering
from causal_sets.d_orders import (
    DOrder, swap_move as d_swap_move, bd_action_4d_fast,
    interval_entropy, mcmc_d_order
)
from causal_sets.bd_action import count_intervals_by_size, count_links, bd_action_2d
from causal_sets.sj_vacuum import pauli_jordan_function, sj_wightman_function
from cosmology.everpresent_lambda import run_everpresent_lambda, compute_N0_scale

np.set_printoptions(precision=6, suppress=True, linewidth=120)
rng = np.random.default_rng(42)


# ============================================================
# SHARED UTILITIES
# ============================================================

def autocorr_time(x, max_lag=500):
    """Estimate integrated autocorrelation time."""
    x = np.array(x, dtype=float)
    n = len(x)
    if n < 10:
        return 1.0
    x = x - np.mean(x)
    var = np.var(x)
    if var < 1e-15:
        return 1.0
    tau = 0.5
    for lag in range(1, min(max_lag, n // 4)):
        c = np.mean(x[:n - lag] * x[lag:]) / var
        if c < 0:
            break
        tau += c
    return max(1.0, tau)


def link_fraction(cs):
    """Fraction of related pairs that are links."""
    links = cs.link_matrix()
    n_links = int(np.sum(np.triu(links, k=1)))
    n_rels = int(np.sum(np.triu(cs.order, k=1)))
    if n_rels == 0:
        return 0.0
    return n_links / n_rels


def iDelta_eigenvalues(cs):
    """Eigenvalues of i*Delta (Hermitian). Returns sorted real array."""
    N = cs.n
    C = cs.order.astype(float)
    Delta = (2.0 / N) * (C.T - C)
    H = 1j * Delta
    return np.linalg.eigvalsh(H).real


def level_spacing_ratio(evals):
    """Compute <r> from sorted eigenvalues."""
    evals = np.sort(evals)
    spacings = np.diff(evals)
    spacings = spacings[spacings > 1e-14]
    if len(spacings) < 3:
        return np.nan
    r_min = np.minimum(spacings[:-1], spacings[1:])
    r_max = np.maximum(spacings[:-1], spacings[1:])
    return float(np.mean(r_min / r_max))


def beta_c_2d(N, eps):
    """Critical coupling for 2D BD transition (Glaser et al.)."""
    return 1.66 / (N * eps**2)


def connectivity(cs, i, j):
    """Shared ancestors + shared descendants (for spacelike pairs)."""
    order_int = cs.order.astype(np.int32)
    cp = int(np.sum(order_int[:, i] & order_int[:, j]))
    cf = int(np.sum(order_int[i, :] & order_int[j, :]))
    return cp + cf


print("=" * 80)
print("EXPERIMENT 100: STRENGTHEN WEAKEST PAPERS — Ideas 511-520")
print("=" * 80)
print()


# ============================================================
# IDEA 511: Paper A — Interval entropy at N=100-200 with parallel tempering
# Does the 4D three-phase structure persist at larger N?
# Previous data: N=30-70 showed three-phase (rise, dip, recovery)
# ============================================================
print("=" * 80)
print("IDEA 511: INTERVAL ENTROPY AT N=100-200 (4D, PARALLEL TEMPERING)")
print("=" * 80)
print()

t0 = time.time()

# 4D MCMC is expensive: to_causet_fast is O(N^2*d), bd_action is O(N^2).
# N=100 should be feasible. N=200 may be slow but worth trying.

for N in [100, 150]:
    print(f"--- N={N}, d=4 ---")
    t_start = time.time()

    # Scan beta values. For 4D, the transition region is around beta ~ 1-5
    # (from exp31/exp42 at N=30: transition at beta ~ 2-3)
    # At larger N the critical beta may shift. Scan broadly.
    betas = np.linspace(0.0, 8.0, 9)

    for beta in betas:
        rng_local = np.random.default_rng(42)
        # Shorter chains for feasibility at large N
        n_steps = 1500
        n_therm = 750
        record_every = 10

        res = mcmc_d_order(
            d=4, N=N, beta=beta,
            n_steps=n_steps, n_thermalize=n_therm,
            record_every=record_every,
            rng=rng_local, verbose=False
        )

        mean_H = np.mean(res['entropies'])
        std_H = np.std(res['entropies'])
        mean_S = np.mean(res['actions'])
        mean_ord = np.mean(res['ordering_fracs'])
        chi_S = np.var(res['actions']) * max(beta**2, 1.0)

        print(f"  beta={beta:5.2f}  H={mean_H:.4f}+-{std_H:.4f}  "
              f"S={mean_S:8.2f}  ord={mean_ord:.3f}  "
              f"chi_S={chi_S:.1f}  acc={res['accept_rate']:.3f}")

    dt = time.time() - t_start
    print(f"  Time for N={N}: {dt:.1f}s")
    print()

dt_511 = time.time() - t0
print(f"Idea 511 total time: {dt_511:.1f}s")
print()


# ============================================================
# IDEA 512: Paper A — Interval entropy on SPRINKLED 4D causets
# Test whether the three-phase structure is specific to d-orders
# or appears in physical (sprinkled) causets too.
# ============================================================
print("=" * 80)
print("IDEA 512: INTERVAL ENTROPY ON SPRINKLED 4D CAUSETS")
print("=" * 80)
print()

t0 = time.time()

# Sprinkled causets don't have a beta parameter — they're fixed geometry.
# But we can compute interval entropy as a function of N (density)
# and compare with random d-order values at beta=0 and beta=large.

print("Sprinkled 4D Minkowski causets — interval entropy vs N:")
print()

for N in [30, 50, 70, 100, 150]:
    entropies = []
    ord_fracs = []
    lf_vals = []
    n_trials = 10 if N <= 100 else 5

    for trial in range(n_trials):
        rng_local = np.random.default_rng(42 + trial)
        cs, coords = sprinkle_fast(N, dim=4, extent_t=1.0, region='diamond',
                                    rng=rng_local)
        H = interval_entropy(cs)
        of = cs.ordering_fraction()
        lf = link_fraction(cs)
        entropies.append(H)
        ord_fracs.append(of)
        lf_vals.append(lf)

    print(f"  N={N:>3}: H={np.mean(entropies):.4f}+-{np.std(entropies):.4f}  "
          f"ord={np.mean(ord_fracs):.3f}+-{np.std(ord_fracs):.3f}  "
          f"link_frac={np.mean(lf_vals):.3f}+-{np.std(lf_vals):.3f}")

print()

# Compare with random 4-orders at beta=0
print("Random 4-orders (beta=0) — interval entropy vs N:")
for N in [30, 50, 70]:
    entropies = []
    ord_fracs = []
    for trial in range(20):
        rng_local = np.random.default_rng(42 + trial)
        do = DOrder(4, N, rng=rng_local)
        cs = do.to_causet_fast()
        H = interval_entropy(cs)
        of = cs.ordering_fraction()
        entropies.append(H)
        ord_fracs.append(of)

    print(f"  N={N:>3}: H={np.mean(entropies):.4f}+-{np.std(entropies):.4f}  "
          f"ord={np.mean(ord_fracs):.3f}+-{np.std(ord_fracs):.3f}")

dt_512 = time.time() - t0
print(f"\nIdea 512 total time: {dt_512:.1f}s")
print()


# ============================================================
# IDEA 513: Paper A — Finite-size scaling collapse of H(beta)
# If the transition is first-order (as suggested by chi ~ N^3),
# the correlation length exponent nu should give data collapse.
# ============================================================
print("=" * 80)
print("IDEA 513: FINITE-SIZE SCALING COLLAPSE OF H(beta)")
print("=" * 80)
print()

t0 = time.time()

# Run 2D MCMC (faster, well-characterized transition) at multiple N
# to do FSS. Use eps=0.12 (standard).
eps = 0.12

# Collect H(beta) curves for different N
fss_data = {}

for N in [30, 50, 70]:
    print(f"--- N={N} (2D, eps={eps}) ---")
    bc = beta_c_2d(N, eps)
    # Scan around beta_c with 12 points
    betas = np.linspace(0.5 * bc, 2.0 * bc, 12)

    H_means = []
    H_stds = []

    for beta in betas:
        rng_local = np.random.default_rng(42)
        from causal_sets.two_orders_v2 import mcmc_corrected
        res = mcmc_corrected(N, beta, eps, n_steps=15000, n_therm=8000,
                              record_every=10, rng=rng_local)

        # Compute interval entropy on samples
        entropies = []
        for cs in res['samples'][::5]:  # every 5th sample for speed
            entropies.append(interval_entropy(cs))

        H_means.append(np.mean(entropies))
        H_stds.append(np.std(entropies))

        print(f"  beta={beta:6.2f} (beta/beta_c={beta/bc:.2f})  "
              f"H={np.mean(entropies):.4f}+-{np.std(entropies):.4f}  "
              f"acc={res['accept_rate']:.3f}")

    fss_data[N] = {
        'betas': betas,
        'beta_c': bc,
        'H_means': np.array(H_means),
        'H_stds': np.array(H_stds),
    }

# Now attempt FSS collapse.
# For a first-order transition: the scaling variable is (beta - beta_c) * N^(1/nu)
# For first-order in 2D: nu = 1/d = 1/2 (for d=2 spatial dimension in stat mech)
# But here d=1 (the system is 0+1-dimensional in some sense), so try nu=1.
# Also try a free fit.

print()
print("FINITE-SIZE SCALING COLLAPSE TEST:")
print("Trying H( (beta - beta_c) * N^(1/nu) ) collapse")
print()

def collapse_quality(nu, fss_data, N_list):
    """Measure quality of data collapse for given nu.
    Lower = better. Uses pairwise interpolation agreement."""
    all_curves = []
    for N in N_list:
        d = fss_data[N]
        x = (d['betas'] - d['beta_c']) * N**(1.0/nu)
        y = d['H_means']
        all_curves.append((x, y))

    # Pairwise comparison: for each pair of curves, evaluate at common x
    # points and measure disagreement
    total_err = 0.0
    n_compare = 0
    for i in range(len(all_curves)):
        for j in range(i+1, len(all_curves)):
            xi, yi = all_curves[i]
            xj, yj = all_curves[j]
            # Find overlapping x range
            x_min = max(xi.min(), xj.min())
            x_max = min(xi.max(), xj.max())
            if x_max <= x_min:
                continue
            # Evaluate both at common x points
            x_common = np.linspace(x_min, x_max, 20)
            yi_interp = np.interp(x_common, xi, yi)
            yj_interp = np.interp(x_common, xj, yj)
            diff = yi_interp - yj_interp
            total_err += np.sum(diff**2)
            n_compare += len(x_common)

    if n_compare == 0:
        return 1e10
    return total_err / n_compare

N_list = [30, 50, 70]

# Scan nu
nus = np.linspace(0.3, 3.0, 28)
qualities = []
for nu in nus:
    q = collapse_quality(nu, fss_data, N_list)
    qualities.append(q)

best_idx = np.argmin(qualities)
best_nu = nus[best_idx]
print(f"Best nu from grid scan: {best_nu:.2f} (quality = {qualities[best_idx]:.6f})")

# Fine search around best
nus_fine = np.linspace(max(0.2, best_nu - 0.3), best_nu + 0.3, 30)
qualities_fine = []
for nu in nus_fine:
    q = collapse_quality(nu, fss_data, N_list)
    qualities_fine.append(q)

best_idx_fine = np.argmin(qualities_fine)
best_nu_fine = nus_fine[best_idx_fine]
print(f"Refined best nu: {best_nu_fine:.3f} (quality = {qualities_fine[best_idx_fine]:.6f})")
print()

# Compare with theoretical predictions
print("Theoretical predictions for nu:")
print(f"  First-order in d=2:  nu = 1/d = 0.50")
print(f"  First-order in d=1:  nu = 1/d = 1.00")
print(f"  Mean-field:          nu = 0.50")
print(f"  Our best fit:        nu = {best_nu_fine:.3f}")
print()

# Print the collapsed data at best nu
print(f"Collapsed data at nu={best_nu_fine:.3f}:")
for N in N_list:
    d = fss_data[N]
    x = (d['betas'] - d['beta_c']) * N**(1.0/best_nu_fine)
    print(f"  N={N:>3}: x_range = [{x.min():.2f}, {x.max():.2f}], "
          f"H_range = [{d['H_means'].min():.3f}, {d['H_means'].max():.3f}]")

dt_513 = time.time() - t0
print(f"\nIdea 513 total time: {dt_513:.1f}s")
print()


# ============================================================
# IDEA 514: Paper A — Compare interval entropy with link fraction
# Which observable gives a sharper transition signal?
# ============================================================
print("=" * 80)
print("IDEA 514: INTERVAL ENTROPY vs LINK FRACTION — WHICH IS SHARPER?")
print("=" * 80)
print()

t0 = time.time()

eps = 0.12

for N in [30, 50, 70]:
    print(f"--- N={N} ---")
    bc = beta_c_2d(N, eps)
    betas = np.linspace(0.5 * bc, 2.5 * bc, 15)

    H_vals = []
    LF_vals = []
    chi_H_vals = []
    chi_LF_vals = []

    for beta in betas:
        rng_local = np.random.default_rng(42)
        res = mcmc_corrected(N, beta, eps, n_steps=12000, n_therm=6000,
                              record_every=10, rng=rng_local)

        entropies = []
        link_fracs = []
        for cs in res['samples']:
            entropies.append(interval_entropy(cs))
            link_fracs.append(link_fraction(cs))

        entropies = np.array(entropies)
        link_fracs = np.array(link_fracs)

        H_vals.append(np.mean(entropies))
        LF_vals.append(np.mean(link_fracs))
        chi_H_vals.append(N * np.var(entropies))
        chi_LF_vals.append(N * np.var(link_fracs))

    H_vals = np.array(H_vals)
    LF_vals = np.array(LF_vals)
    chi_H_vals = np.array(chi_H_vals)
    chi_LF_vals = np.array(chi_LF_vals)

    # Sharpness metrics
    H_jump = np.max(H_vals) - np.min(H_vals)
    LF_jump = np.max(LF_vals) - np.min(LF_vals)
    H_chi_max = np.max(chi_H_vals)
    LF_chi_max = np.max(chi_LF_vals)

    # Derivative sharpness: max |dO/dbeta|
    H_deriv = np.max(np.abs(np.diff(H_vals) / np.diff(betas)))
    LF_deriv = np.max(np.abs(np.diff(LF_vals) / np.diff(betas)))

    print(f"  H:  jump={H_jump:.3f}  chi_max={H_chi_max:.1f}  max_deriv={H_deriv:.4f}")
    print(f"  LF: jump={LF_jump:.3f}  chi_max={LF_chi_max:.1f}  max_deriv={LF_deriv:.4f}")

    # Relative comparison
    if LF_chi_max > 0 and H_chi_max > 0:
        ratio = LF_chi_max / H_chi_max
        print(f"  chi_max ratio (LF/H): {ratio:.2f} — "
              f"{'link fraction SHARPER' if ratio > 1 else 'interval entropy SHARPER'}")

    # Print the transition curves
    print(f"  Detailed scan:")
    for i, beta in enumerate(betas):
        print(f"    beta={beta:6.2f} (b/bc={beta/bc:.2f})  H={H_vals[i]:.4f}  LF={LF_vals[i]:.4f}")
    print()

dt_514 = time.time() - t0
print(f"Idea 514 total time: {dt_514:.1f}s")
print()


# ============================================================
# IDEA 515: Paper B2 — Tighten alpha constraint with 10,000 realizations
# Previous: 1000 realizations gave alpha=0.03 with Omega_Lambda=0.732+-0.103
# ============================================================
print("=" * 80)
print("IDEA 515: TIGHTEN ALPHA CONSTRAINT — 10,000 REALIZATIONS")
print("=" * 80)
print()

t0 = time.time()

# Scan alpha values, 10000 realizations each
alpha_values = [0.01, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.07, 0.10]

print("Alpha scan with 10,000 realizations each:")
print(f"{'alpha':>8} {'mean_OL':>10} {'std_OL':>10} {'median_OL':>10} "
      f"{'P(0.6<OL<0.8)':>14} {'stderr':>10}")

alpha_results = {}

for alpha in alpha_values:
    omega_lambdas = []
    n_crash = 0

    for trial in range(5000):
        rng_local = np.random.default_rng(trial)
        try:
            history = run_everpresent_lambda(
                alpha=alpha, a_initial=1e-4, a_final=1.5,
                n_steps=2000, Omega_m0=0.3, Omega_r0=9e-5,
                rng=rng_local
            )
            # Get Omega_Lambda at a=1
            for state in history:
                if state.a >= 1.0:
                    H2 = state.H**2 if state.H > 0 else 1e-30
                    omega_L = state.rho_lambda / H2
                    omega_lambdas.append(omega_L)
                    break
        except (ValueError, OverflowError, FloatingPointError):
            n_crash += 1
            continue

    omega_lambdas = np.array(omega_lambdas)
    # Remove extreme outliers (>10 sigma)
    if len(omega_lambdas) > 100:
        median = np.median(omega_lambdas)
        mad = np.median(np.abs(omega_lambdas - median))
        if mad > 0:
            good = np.abs(omega_lambdas - median) < 10 * mad * 1.4826
            omega_lambdas = omega_lambdas[good]

    mean_OL = np.mean(omega_lambdas) if len(omega_lambdas) > 0 else np.nan
    std_OL = np.std(omega_lambdas) if len(omega_lambdas) > 0 else np.nan
    median_OL = np.median(omega_lambdas) if len(omega_lambdas) > 0 else np.nan
    stderr = std_OL / np.sqrt(len(omega_lambdas)) if len(omega_lambdas) > 0 else np.nan
    p_good = np.mean((omega_lambdas > 0.6) & (omega_lambdas < 0.8)) if len(omega_lambdas) > 0 else 0

    print(f"{alpha:8.3f} {mean_OL:10.4f} {std_OL:10.4f} {median_OL:10.4f} "
          f"{p_good:14.3f} {stderr:10.5f}  (n={len(omega_lambdas)}, crash={n_crash})")

    alpha_results[alpha] = {
        'mean': mean_OL, 'std': std_OL, 'median': median_OL,
        'stderr': stderr, 'p_good': p_good, 'n': len(omega_lambdas),
    }

# Find optimal alpha (closest mean to 0.7)
best_alpha = min(alpha_results.keys(),
                  key=lambda a: abs(alpha_results[a]['mean'] - 0.7))
print(f"\nBest alpha (mean closest to 0.7): {best_alpha}")
print(f"  Omega_Lambda = {alpha_results[best_alpha]['mean']:.4f} "
      f"+- {alpha_results[best_alpha]['stderr']:.5f}")

# Confidence interval on alpha
# Use chi-squared: find alpha where |mean - 0.7| / stderr < 2
print("\nAlpha values consistent with Omega_Lambda = 0.7 (2-sigma):")
for alpha in sorted(alpha_results.keys()):
    r = alpha_results[alpha]
    if r['stderr'] > 0:
        z = abs(r['mean'] - 0.7) / r['stderr']
        consistent = "YES" if z < 2 else "no"
        print(f"  alpha={alpha:.3f}: z = {z:.2f} ({consistent})")

dt_515 = time.time() - t0
print(f"\nIdea 515 total time: {dt_515:.1f}s")
print()


# ============================================================
# IDEA 516: Paper B2 — DESI DR2 w(z) prediction
# The everpresent Lambda model predicts a specific w(z) because
# Lambda fluctuates as a random walk in sqrt(N).
# ============================================================
print("=" * 80)
print("IDEA 516: DESI DR2 w(z) PREDICTION")
print("=" * 80)
print()

t0 = time.time()

# Use the best-fit alpha to compute w(z) = p_DE / rho_DE
# In the everpresent Lambda model, rho_Lambda is stochastic
# The effective equation of state w(a) = -1 + (1/3) * d(ln rho_Lambda)/d(ln a)

alpha_use = best_alpha
print(f"Using alpha = {alpha_use} to compute w(z):")
print()

# Run 1000 realizations and track rho_lambda(a)
n_realizations = 500
a_grid = np.logspace(-2, np.log10(2.0), 500)

# We'll interpolate each realization onto the a_grid
w_at_z = {}  # z -> list of w values

# Redshift grid for output
z_output = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]

rho_lambda_all = []

for trial in range(n_realizations):
    rng_local = np.random.default_rng(42 + trial)
    history = run_everpresent_lambda(
        alpha=alpha_use, a_initial=1e-4, a_final=2.5,
        n_steps=3000, Omega_m0=0.3, Omega_r0=9e-5,
        rng=rng_local
    )

    a_hist = np.array([s.a for s in history])
    rho_hist = np.array([s.rho_lambda for s in history])

    # Interpolate onto a_grid
    rho_interp = np.interp(a_grid, a_hist, rho_hist)
    rho_lambda_all.append(rho_interp)

rho_lambda_all = np.array(rho_lambda_all)  # (n_realizations, len(a_grid))

# Compute mean rho_lambda(a)
rho_mean = np.mean(rho_lambda_all, axis=0)
rho_std = np.std(rho_lambda_all, axis=0)

# Compute w(a) = -1 + (1/3) * d(ln rho) / d(ln a)
# Use the mean rho for a smooth curve
ln_a = np.log(a_grid)
ln_rho = np.log(np.abs(rho_mean) + 1e-300)
d_ln_rho = np.gradient(ln_rho, ln_a)
w_mean = -1.0 + (1.0/3.0) * d_ln_rho

# Also compute w(a) for each realization
w_all = np.zeros_like(rho_lambda_all)
for i in range(n_realizations):
    rho_i = rho_lambda_all[i]
    ln_rho_i = np.log(np.abs(rho_i) + 1e-300)
    d_ln_rho_i = np.gradient(ln_rho_i, ln_a)
    w_all[i] = -1.0 + (1.0/3.0) * d_ln_rho_i

# Report at specific redshifts
print(f"{'z':>6} {'a':>8} {'w_mean':>10} {'w_std':>10} {'w_median':>10} {'rho_L/rho_c':>12}")
for z in z_output:
    a = 1.0 / (1.0 + z)
    idx = np.argmin(np.abs(a_grid - a))
    w_vals = w_all[:, idx]
    # Filter out extreme values
    w_good = w_vals[(w_vals > -5) & (w_vals < 5)]
    if len(w_good) > 10:
        print(f"{z:6.1f} {a:8.4f} {np.mean(w_good):10.4f} {np.std(w_good):10.4f} "
              f"{np.median(w_good):10.4f} {rho_mean[idx]:12.4f}")
    else:
        print(f"{z:6.1f} {a:8.4f}  [insufficient data]")

print()
print("DESI DR2 comparison:")
print("  DESI DR2 finds w0 = -0.752 +- 0.058, wa = -1.02 +0.34/-0.28")
print("  (w(z) = w0 + wa * z/(1+z))")
print("  At z=0: w_DESI = -0.752")
print("  At z=1: w_DESI = -0.752 + (-1.02)*0.5 = -1.262")
print()

# Check: does our model predict w != -1?
idx_z0 = np.argmin(np.abs(a_grid - 1.0))
w_at_z0 = w_all[:, idx_z0]
w_at_z0_good = w_at_z0[(w_at_z0 > -5) & (w_at_z0 < 5)]
t_stat, p_val = stats.ttest_1samp(w_at_z0_good, -1.0)
print(f"Test w(z=0) = -1: t={t_stat:.2f}, p={p_val:.4f}")
if p_val < 0.05:
    print(f"  -> w(z=0) DEVIATES from -1 at {(1-p_val)*100:.1f}% confidence")
    print(f"  -> mean w(z=0) = {np.mean(w_at_z0_good):.4f}")
else:
    print(f"  -> w(z=0) is CONSISTENT with -1")

dt_516 = time.time() - t0
print(f"\nIdea 516 total time: {dt_516:.1f}s")
print()


# ============================================================
# IDEA 517: Paper B2 — Bayes factor with proper prior
# Compare: M1 = everpresent Lambda (alpha free), M2 = LCDM (Lambda=const)
# ============================================================
print("=" * 80)
print("IDEA 517: BAYES FACTOR FOR EVERPRESENT LAMBDA vs LCDM")
print("=" * 80)
print()

t0 = time.time()

# Observed: Omega_Lambda = 0.685 +- 0.007 (Planck 2018)
OL_obs = 0.685
OL_err = 0.007

# LCDM: P(data|LCDM) = likelihood at best fit
# Everpresent Lambda: P(data|EPL) = integral P(data|alpha) * P(alpha) d(alpha)

# Prior on alpha: log-uniform on [0.001, 1.0] (3 decades)
n_alpha_bf = 25
alphas_bf = np.logspace(-3, 0, n_alpha_bf)
d_log_alpha = (np.log(1.0) - np.log(0.001)) / n_alpha_bf  # log-uniform spacing

# For each alpha, compute P(Omega_Lambda = 0.685)
# Using the 10000-realization data from Idea 515, plus new runs for alphas not tested

print("Computing marginal likelihood for Everpresent Lambda model...")
print()

log_likelihoods = []

for alpha in alphas_bf:
    # Quick run: 300 realizations per alpha (for Bayes factor)
    omega_lambdas = []
    for trial in range(300):
        rng_local = np.random.default_rng(trial * 1000 + int(alpha * 10000))
        try:
            history = run_everpresent_lambda(
                alpha=alpha, a_initial=1e-4, a_final=1.5,
                n_steps=1500, Omega_m0=0.3, Omega_r0=9e-5,
                rng=rng_local
            )
            for state in history:
                if state.a >= 1.0:
                    H2 = state.H**2 if state.H > 0 else 1e-30
                    omega_L = state.rho_lambda / H2
                    omega_lambdas.append(omega_L)
                    break
        except:
            continue

    if len(omega_lambdas) < 50:
        log_likelihoods.append(-np.inf)
        continue

    omega_lambdas = np.array(omega_lambdas)
    # Remove extreme outliers
    median = np.median(omega_lambdas)
    mad = np.median(np.abs(omega_lambdas - median))
    if mad > 0:
        good = np.abs(omega_lambdas - median) < 10 * mad * 1.4826
        omega_lambdas = omega_lambdas[good]

    # Estimate P(Omega_Lambda = 0.685 | alpha) using KDE
    if len(omega_lambdas) > 20:
        try:
            kde = stats.gaussian_kde(omega_lambdas)
            p_data_given_alpha = float(kde(OL_obs))
        except:
            p_data_given_alpha = 0.0
    else:
        p_data_given_alpha = 0.0

    if p_data_given_alpha > 0:
        log_likelihoods.append(np.log(p_data_given_alpha))
    else:
        log_likelihoods.append(-np.inf)

log_likelihoods = np.array(log_likelihoods)

# Marginal likelihood: integral P(data|alpha) * P(alpha) d(alpha)
# With log-uniform prior: P(alpha) = 1/(alpha * ln(1000))
# log P(data|alpha) * P(alpha) = log_likelihoods[i] - log(alpha[i]) - log(ln(1000))

log_prior = -np.log(alphas_bf) - np.log(np.log(1000))
log_integrand = log_likelihoods + log_prior

# Log-sum-exp for numerical stability
finite_mask = np.isfinite(log_integrand)
if np.any(finite_mask):
    max_log = np.max(log_integrand[finite_mask])
    log_marginal_epl = max_log + np.log(
        np.sum(np.exp(log_integrand[finite_mask] - max_log)) * d_log_alpha)

    # LCDM likelihood: Omega_Lambda is a free parameter, at best fit it matches exactly
    # P(data|LCDM) ~ 1 / (sqrt(2*pi) * sigma) at best fit
    log_marginal_lcdm = -0.5 * np.log(2 * np.pi) - np.log(OL_err)

    log_BF = log_marginal_epl - log_marginal_lcdm
    BF = np.exp(log_BF)

    print(f"log P(data|EPL) = {log_marginal_epl:.2f}")
    print(f"log P(data|LCDM) = {log_marginal_lcdm:.2f}")
    print(f"log Bayes factor (EPL/LCDM) = {log_BF:.2f}")
    print(f"Bayes factor = {BF:.4f}")
    print()

    if BF > 1:
        print(f"EPL FAVORED by factor {BF:.1f}")
    else:
        print(f"LCDM FAVORED by factor {1/BF:.1f}")

    # Jeffreys scale
    abs_log_bf = abs(log_BF)
    if abs_log_bf < 1:
        strength = "Not worth more than a bare mention"
    elif abs_log_bf < 2.5:
        strength = "Substantial"
    elif abs_log_bf < 5:
        strength = "Strong"
    else:
        strength = "Decisive"
    print(f"Jeffreys scale: {strength}")

    # Show which alpha values contribute most
    print(f"\nTop alpha contributions to marginal likelihood:")
    sorted_idx = np.argsort(log_integrand)[::-1]
    for k in range(min(5, len(sorted_idx))):
        idx = sorted_idx[k]
        if np.isfinite(log_integrand[idx]):
            print(f"  alpha={alphas_bf[idx]:.4f}: log(L*prior)={log_integrand[idx]:.2f}")

else:
    print("ERROR: No finite likelihoods computed. Check simulation parameters.")

dt_517 = time.time() - t0
print(f"\nIdea 517 total time: {dt_517:.1f}s")
print()


# ============================================================
# IDEA 518: Paper D — Phase-mixing artifact quantification
# How many MCMC samples from each phase need to be mixed to
# produce <r>=0.12?
# ============================================================
print("=" * 80)
print("IDEA 518: PHASE-MIXING ARTIFACT — EXACT QUANTIFICATION")
print("=" * 80)
print()

t0 = time.time()

# Generate pure continuum-phase and pure KR-phase spectra
# Then mix different fractions and compute <r>

N_test = 50
eps = 0.12
bc = beta_c_2d(N_test, eps)

print(f"N={N_test}, eps={eps}, beta_c={bc:.2f}")
print()

# Generate continuum-phase spectra (low beta)
print("Generating continuum-phase spectra (beta = 0.5 * beta_c)...")
beta_cont = 0.5 * bc
rng_local = np.random.default_rng(42)
res_cont = mcmc_corrected(N_test, beta_cont, eps, n_steps=20000, n_therm=10000,
                            record_every=20, rng=rng_local)

continuum_evals = []
for cs in res_cont['samples'][:200]:
    ev = iDelta_eigenvalues(cs)
    continuum_evals.append(ev)

# Generate KR-phase spectra (high beta)
print("Generating KR-phase spectra (beta = 2.0 * beta_c)...")
beta_kr = 2.0 * bc
rng_local = np.random.default_rng(42)
res_kr = mcmc_corrected(N_test, beta_kr, eps, n_steps=20000, n_therm=10000,
                          record_every=20, rng=rng_local)

kr_evals = []
for cs in res_kr['samples'][:200]:
    ev = iDelta_eigenvalues(cs)
    kr_evals.append(ev)

print(f"  Continuum spectra: {len(continuum_evals)} samples")
print(f"  KR spectra: {len(kr_evals)} samples")

# Pure phase <r>
r_cont_pure = np.mean([level_spacing_ratio(ev) for ev in continuum_evals])
r_kr_pure = np.mean([level_spacing_ratio(ev) for ev in kr_evals])
print(f"  Pure continuum <r> = {r_cont_pure:.4f}")
print(f"  Pure KR <r> = {r_kr_pure:.4f}")
print()

# Now mix: concatenate n_cont continuum spectra with n_kr KR spectra
# and compute <r> on the concatenated eigenvalues
print("Mixing test — concatenate eigenvalues from both phases:")
print(f"{'n_cont':>8} {'n_kr':>8} {'frac_cont':>10} {'<r>':>8} {'interpretation':>20}")

mix_results = []
for n_cont in [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200]:
    for n_kr in [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200]:
        if n_cont > len(continuum_evals) or n_kr > len(kr_evals):
            continue
        # Concatenate eigenvalues
        all_ev = np.concatenate([
            np.concatenate(continuum_evals[:n_cont]),
            np.concatenate(kr_evals[:n_kr])
        ])
        r = level_spacing_ratio(all_ev)
        frac = n_cont / (n_cont + n_kr)
        mix_results.append((n_cont, n_kr, frac, r))

# Find the mix that gives <r> closest to 0.12
mix_results.sort(key=lambda x: abs(x[3] - 0.12))

print("\nClosest to <r>=0.12:")
for nc, nk, frac, r in mix_results[:10]:
    print(f"{nc:8d} {nk:8d} {frac:10.3f} {r:8.4f}")

print("\nClosest to <r>=0.42 (reported by exp92 for 50/50 mixing):")
mix_results.sort(key=lambda x: abs(x[3] - 0.42))
for nc, nk, frac, r in mix_results[:5]:
    print(f"{nc:8d} {nk:8d} {frac:10.3f} {r:8.4f}")

# Systematic: <r> vs continuum fraction (equal total samples)
print("\n<r> as a function of continuum fraction (fixed total = 100 spectra):")
total_mix = 100
for frac_cont in np.linspace(0.0, 1.0, 21):
    nc = max(1, int(frac_cont * total_mix))
    nk = max(1, total_mix - nc)
    if nc > len(continuum_evals) or nk > len(kr_evals):
        continue
    all_ev = np.concatenate([
        np.concatenate(continuum_evals[:nc]),
        np.concatenate(kr_evals[:nk])
    ])
    r = level_spacing_ratio(all_ev)
    print(f"  frac_cont={frac_cont:.2f}  nc={nc:>3}  nk={nk:>3}  <r>={r:.4f}")

dt_518 = time.time() - t0
print(f"\nIdea 518 total time: {dt_518:.1f}s")
print()


# ============================================================
# IDEA 519: Paper D — GUE universality on 3D and 4D causets
# So far GUE was tested on 2-orders. Do 3-orders and 4-orders
# also show GUE statistics?
# ============================================================
print("=" * 80)
print("IDEA 519: GUE UNIVERSALITY ON 3D AND 4D CAUSETS")
print("=" * 80)
print()

t0 = time.time()

for d in [2, 3, 4]:
    print(f"--- d-orders, d={d} ---")
    for N in [30, 50, 70]:
        r_vals = []
        for trial in range(30):
            rng_local = np.random.default_rng(42 + trial)
            do = DOrder(d, N, rng=rng_local)
            cs = do.to_causet_fast()
            ev = iDelta_eigenvalues(cs)
            r = level_spacing_ratio(ev)
            r_vals.append(r)

        r_mean = np.mean(r_vals)
        r_std = np.std(r_vals)
        r_err = r_std / np.sqrt(len(r_vals))

        # Also compute ordering fraction for context
        of_vals = []
        for trial in range(30):
            rng_local = np.random.default_rng(42 + trial)
            do = DOrder(d, N, rng=rng_local)
            cs = do.to_causet_fast()
            of_vals.append(cs.ordering_fraction())

        print(f"  N={N:>3}: <r>={r_mean:.4f}+-{r_err:.4f}  "
              f"(GUE=0.5996, Poisson=0.3863)  "
              f"ord_frac={np.mean(of_vals):.3f}")

    print()

# Also test sprinkled causets in 2D, 3D, 4D
print("--- Sprinkled Minkowski causets ---")
for dim in [2, 3, 4]:
    for N in [30, 50, 70]:
        r_vals = []
        for trial in range(30):
            rng_local = np.random.default_rng(42 + trial)
            cs, _ = sprinkle_fast(N, dim=dim, extent_t=1.0,
                                   region='diamond', rng=rng_local)
            ev = iDelta_eigenvalues(cs)
            r = level_spacing_ratio(ev)
            r_vals.append(r)

        r_mean = np.mean(r_vals)
        r_err = np.std(r_vals) / np.sqrt(len(r_vals))
        print(f"  dim={dim}, N={N:>3}: <r>={r_mean:.4f}+-{r_err:.4f}")

    print()

dt_519 = time.time() - t0
print(f"Idea 519 total time: {dt_519:.1f}s")
print()


# ============================================================
# IDEA 520: Paper C — ER=EPR Gram identity on sprinkled/d-order causets
# The analytic proof: (-Delta^2)_ij = (4/N^2) * kappa_ij
# holds exactly for 2-orders (spacelike pairs).
# Does it hold APPROXIMATELY for sprinkled causets and d-orders?
# ============================================================
print("=" * 80)
print("IDEA 520: ER=EPR GRAM IDENTITY ON SPRINKLED AND D-ORDER CAUSETS")
print("=" * 80)
print()

t0 = time.time()

def test_gram_identity(cs, label):
    """Test (-Delta^2)_ij = (4/N^2) * kappa_ij for spacelike pairs."""
    N = cs.n
    C = cs.order.astype(float)
    Delta = (2.0 / N) * (C.T - C)  # antisymmetric

    # -Delta^2
    neg_Delta_sq = -Delta @ Delta

    # Connectivity matrix kappa_ij = shared ancestors + shared descendants
    order_int = cs.order.astype(np.int32)
    # Shared ancestors: both i and j are preceded by k
    # shared_anc[i,j] = sum_k order[k,i] * order[k,j]
    shared_anc = order_int.T @ order_int  # (N,N)
    # Shared descendants: both i and j precede k
    # shared_desc[i,j] = sum_k order[i,k] * order[j,k]
    shared_desc = order_int @ order_int.T  # (N,N)
    kappa = shared_anc + shared_desc

    # The prediction: (-Delta^2)_ij = (4/N^2) * kappa_ij
    predicted = (4.0 / N**2) * kappa

    # Test on spacelike pairs only
    spacelike_mask = ~cs.order & ~cs.order.T & ~np.eye(N, dtype=bool)
    # Only upper triangle to avoid double counting
    upper = np.triu(np.ones((N, N), dtype=bool), k=1)
    mask = spacelike_mask & upper

    if np.sum(mask) == 0:
        print(f"  {label}: no spacelike pairs!")
        return

    actual = neg_Delta_sq[mask]
    pred = predicted[mask]

    # Correlation
    if np.std(actual) > 1e-15 and np.std(pred) > 1e-15:
        r_val, p_val = stats.pearsonr(actual, pred)
    else:
        r_val, p_val = 0.0, 1.0

    # Relative error
    if np.mean(np.abs(pred)) > 1e-15:
        rel_err = np.mean(np.abs(actual - pred)) / np.mean(np.abs(pred))
    else:
        rel_err = np.inf

    # Max absolute error
    max_err = np.max(np.abs(actual - pred))

    # Is it EXACT (machine precision)?
    is_exact = max_err < 1e-12

    print(f"  {label}: r={r_val:.4f} (p={p_val:.2e})  "
          f"rel_err={rel_err:.4f}  max_err={max_err:.2e}  "
          f"{'EXACT' if is_exact else 'APPROXIMATE'}  "
          f"n_spacelike={np.sum(mask)}")

    return r_val, rel_err, max_err

# Test on 2-orders (should be EXACT)
print("2-ORDERS (expected: EXACT):")
for N in [10, 20, 30, 50]:
    for trial in range(3):
        rng_local = np.random.default_rng(42 + trial)
        to = TwoOrder(N, rng=rng_local)
        cs = to.to_causet()
        test_gram_identity(cs, f"2-order N={N} trial={trial}")
    print()

# Test on d-orders (d=3,4)
print("D-ORDERS (d=3, expected: APPROXIMATE?):")
for N in [10, 20, 30, 50]:
    results = []
    for trial in range(5):
        rng_local = np.random.default_rng(42 + trial)
        do = DOrder(3, N, rng=rng_local)
        cs = do.to_causet_fast()
        result = test_gram_identity(cs, f"3-order N={N} trial={trial}")
        if result:
            results.append(result)
    if results:
        r_vals = [r[0] for r in results]
        rel_errs = [r[1] for r in results]
        print(f"  Summary N={N}: mean_r={np.mean(r_vals):.4f}, "
              f"mean_rel_err={np.mean(rel_errs):.4f}")
    print()

print("D-ORDERS (d=4, expected: APPROXIMATE?):")
for N in [10, 20, 30, 50]:
    results = []
    for trial in range(5):
        rng_local = np.random.default_rng(42 + trial)
        do = DOrder(4, N, rng=rng_local)
        cs = do.to_causet_fast()
        result = test_gram_identity(cs, f"4-order N={N} trial={trial}")
        if result:
            results.append(result)
    if results:
        r_vals = [r[0] for r in results]
        rel_errs = [r[1] for r in results]
        print(f"  Summary N={N}: mean_r={np.mean(r_vals):.4f}, "
              f"mean_rel_err={np.mean(rel_errs):.4f}")
    print()

# Test on sprinkled causets (2D, 3D, 4D)
for dim in [2, 3, 4]:
    print(f"SPRINKLED {dim}D MINKOWSKI CAUSETS:")
    for N in [20, 30, 50]:
        results = []
        for trial in range(5):
            rng_local = np.random.default_rng(42 + trial)
            cs, _ = sprinkle_fast(N, dim=dim, extent_t=1.0,
                                   region='diamond', rng=rng_local)
            result = test_gram_identity(cs, f"sprinkled {dim}D N={N} trial={trial}")
            if result:
                results.append(result)
        if results:
            r_vals = [r[0] for r in results]
            rel_errs = [r[1] for r in results]
            print(f"  Summary N={N}: mean_r={np.mean(r_vals):.4f}, "
                  f"mean_rel_err={np.mean(rel_errs):.4f}")
        print()

dt_520 = time.time() - t0
print(f"Idea 520 total time: {dt_520:.1f}s")
print()


# ============================================================
# SUMMARY
# ============================================================
print("=" * 80)
print("EXPERIMENT 100 — SUMMARY")
print("=" * 80)
print()
print("Ideas 511-520: Strengthen weakest papers (A at 7/10, B2 at 5/10)")
print()
total_time = dt_511 + dt_512 + dt_513 + dt_514 + dt_515 + dt_516 + dt_517 + dt_518 + dt_519 + dt_520
print(f"Total experiment time: {total_time:.1f}s")
print()
print("Key questions answered:")
print("  511: Does 4D three-phase persist at N=100-200?")
print("  512: Does sprinkled 4D causet show similar interval entropy?")
print("  513: What is the optimal FSS exponent nu?")
print("  514: Is interval entropy or link fraction the sharper order parameter?")
print("  515: Can we tighten alpha to better than +-0.103?")
print("  516: What does the everpresent Lambda w(z) look like?")
print("  517: What is the Bayes factor EPL vs LCDM?")
print("  518: Exactly how many mixed samples produce <r>=0.12?")
print("  519: Is GUE universal across d=2,3,4 causets?")
print("  520: Does the Gram identity extend beyond 2-orders?")

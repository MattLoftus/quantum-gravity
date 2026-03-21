"""
Experiment 92: STRENGTHENING PAPER D — SPECTRAL STATISTICS AT THE TRANSITION
Ideas 431-440

The sub-Poisson <r>=0.12 is NOT a fundamental KR property — it's a TRANSITION
phenomenon. Pure KR has <r>=0.54 (GUE-like). The sub-Poisson dip occurs when
the system is caught between phases. This experiment characterizes that dip
in detail to push Paper D's score from 7 toward 7.5.

Ideas:
431. Fine beta scan near beta_c: WHEN does <r> reach minimum? (20 pts, N=50)
432. Does the minimum <r> DEEPEN with N? (N=30,50,70, fine scan)
433. MCMC dynamics: how many steps after crossing beta_c to reach min <r>?
434. Full P(s) distribution at the transition — mixture of two distributions?
435. Correlation between sub-Poisson <r> and ACTION BIMODALITY
436. Mixture test: blend continuum + KR configs (50/50) — does mixing produce sub-Poisson?
437. Level spacing ratio of WIGHTMAN function W (not just iDelta)
438. <r> at the 4D BD transition — same sub-Poisson dip?
439. Epsilon dependence: dip deeper at eps=0.05 (sharper transition) vs eps=0.25?
440. Partially ordered total order: <r> for chain with random relations removed
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats
import time

from causal_sets.two_orders import TwoOrder, swap_move
from causal_sets.two_orders_v2 import bd_action_corrected
from causal_sets.fast_core import FastCausalSet, sprinkle_fast
from causal_sets.bd_action import count_intervals_by_size, bd_action_4d
from causal_sets.sj_vacuum import pauli_jordan_function, sj_wightman_function

np.set_printoptions(precision=6, suppress=True, linewidth=120)
rng = np.random.default_rng(42)


# ============================================================
# SHARED UTILITIES
# ============================================================

def beta_c(N, eps):
    """Critical coupling from Glaser et al."""
    return 1.66 / (N * eps**2)


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


def spacing_distribution(evals):
    """Return normalized spacings s_n / <s> for P(s) histogram."""
    evals = np.sort(evals)
    spacings = np.diff(evals)
    spacings = spacings[spacings > 1e-14]
    if len(spacings) < 2:
        return np.array([])
    mean_s = np.mean(spacings)
    if mean_s < 1e-14:
        return np.array([])
    return spacings / mean_s


def run_mcmc_with_tracking(N, beta, eps, n_steps, n_therm, record_every,
                           rng_local, track_r=False, track_every=100):
    """MCMC loop with optional per-step <r> tracking."""
    current = TwoOrder(N, rng=rng_local)
    current_cs = current.to_causet()
    current_S = bd_action_corrected(current_cs, eps)
    n_acc = 0

    samples = []
    actions = []
    r_track = []  # (step, r_value) pairs

    for step in range(n_steps):
        proposed = swap_move(current, rng_local)
        proposed_cs = proposed.to_causet()
        proposed_S = bd_action_corrected(proposed_cs, eps)

        dS = beta * (proposed_S - current_S)
        if dS <= 0 or rng_local.random() < np.exp(-dS):
            current = proposed
            current_cs = proposed_cs
            current_S = proposed_S
            n_acc += 1

        if step >= n_therm and (step - n_therm) % record_every == 0:
            samples.append(current_cs)
            actions.append(current_S)

        if track_r and step % track_every == 0:
            evals = iDelta_eigenvalues(current_cs)
            r_val = level_spacing_ratio(evals)
            r_track.append((step, r_val))

    return {
        'samples': samples,
        'actions': np.array(actions),
        'accept_rate': n_acc / n_steps,
        'r_track': r_track,
    }


print("=" * 78)
print("EXPERIMENT 92: STRENGTHENING PAPER D — SPECTRAL STATISTICS AT TRANSITION")
print("Ideas 431-440")
print("=" * 78)
print()


# ============================================================
# IDEA 431: FINE BETA SCAN — WHERE DOES <r> REACH ITS MINIMUM?
# ============================================================

print("=" * 78)
print("IDEA 431: FINE BETA SCAN NEAR beta_c")
print("20 points from 0.8*beta_c to 4*beta_c, N=50, eps=0.12")
print("=" * 78)
print()

t0 = time.time()
N_431 = 50
EPS_431 = 0.12
bc_431 = beta_c(N_431, EPS_431)

# 20 points from 0.8*bc to 4*bc
beta_multiples_431 = np.linspace(0.8, 4.0, 20)
betas_431 = beta_multiples_431 * bc_431

results_431 = {}  # beta_mult -> (mean_r, std_r, mean_action)

for i, (bm, beta) in enumerate(zip(beta_multiples_431, betas_431)):
    r_values = []
    action_values = []
    # 3 independent runs for statistics
    for run in range(3):
        res = run_mcmc_with_tracking(
            N_431, beta, EPS_431, n_steps=12000, n_therm=8000,
            record_every=200, rng_local=np.random.default_rng(42 + run*100 + i)
        )
        for cs in res['samples']:
            evals = iDelta_eigenvalues(cs)
            r_values.append(level_spacing_ratio(evals))
        action_values.extend(res['actions'].tolist())

    r_arr = np.array([r for r in r_values if not np.isnan(r)])
    a_arr = np.array(action_values)
    results_431[bm] = (np.mean(r_arr), np.std(r_arr), np.mean(a_arr))
    print(f"  beta/bc={bm:.2f}  beta={beta:.2f}  <r>={np.mean(r_arr):.4f} +/- {np.std(r_arr):.4f}  "
          f"<S>/N={np.mean(a_arr)/N_431:.4f}  ({len(r_arr)} samples)")

# Find minimum
min_bm = min(results_431.keys(), key=lambda k: results_431[k][0])
print(f"\n  MINIMUM <r> at beta/bc = {min_bm:.2f} (beta = {min_bm*bc_431:.2f})")
print(f"    <r> = {results_431[min_bm][0]:.4f} +/- {results_431[min_bm][1]:.4f}")
print(f"    <S>/N = {results_431[min_bm][2]/N_431:.4f}")
print(f"  Time: {time.time()-t0:.1f}s")
print()


# ============================================================
# IDEA 432: DOES THE MINIMUM <r> DEEPEN WITH N?
# ============================================================

print("=" * 78)
print("IDEA 432: DOES THE MIN <r> DEEPEN WITH N?")
print("N=30,50,70 with fine beta scan around the dip")
print("=" * 78)
print()

t0 = time.time()
EPS_432 = 0.12
Ns_432 = [30, 50, 70]

for N_val in Ns_432:
    bc_val = beta_c(N_val, EPS_432)
    # Scan 10 points near the transition
    beta_mults = np.linspace(0.8, 3.5, 10)
    betas_scan = beta_mults * bc_val

    # Adjust MCMC parameters with N
    n_steps_val = max(10000, N_val * 200)
    n_therm_val = n_steps_val // 2

    min_r = 1.0
    min_bm_val = 0
    all_r = []

    for bm, beta in zip(beta_mults, betas_scan):
        r_values = []
        for run in range(2):
            res = run_mcmc_with_tracking(
                N_val, beta, EPS_432, n_steps=n_steps_val, n_therm=n_therm_val,
                record_every=max(100, n_steps_val // 40),
                rng_local=np.random.default_rng(42 + run*100 + N_val)
            )
            for cs in res['samples']:
                evals = iDelta_eigenvalues(cs)
                r_values.append(level_spacing_ratio(evals))

        r_arr = np.array([r for r in r_values if not np.isnan(r)])
        mean_r = np.mean(r_arr) if len(r_arr) > 0 else np.nan
        all_r.append((bm, mean_r, np.std(r_arr) if len(r_arr) > 0 else np.nan))
        if mean_r < min_r:
            min_r = mean_r
            min_bm_val = bm

    print(f"  N={N_val}, beta_c={bc_val:.2f}:")
    for bm, mr, sr in all_r:
        print(f"    beta/bc={bm:.2f}  <r>={mr:.4f} +/- {sr:.4f}")
    print(f"    MIN <r> = {min_r:.4f} at beta/bc = {min_bm_val:.2f}")
    print()

print(f"  Time: {time.time()-t0:.1f}s")
print()


# ============================================================
# IDEA 433: MCMC DYNAMICS — HOW MANY STEPS TO REACH MIN <r>?
# ============================================================

print("=" * 78)
print("IDEA 433: MCMC DYNAMICS — <r> vs MCMC step")
print("Start from random config, quench to 2*beta_c, track <r> every 200 steps")
print("=" * 78)
print()

t0 = time.time()
N_433 = 50
EPS_433 = 0.12
bc_433 = beta_c(N_433, EPS_433)
beta_quench = 2.0 * bc_433

# Run with tracking (no thermalization — we want to see the dynamics)
res_433 = run_mcmc_with_tracking(
    N_433, beta_quench, EPS_433,
    n_steps=20000, n_therm=20000,  # no samples, just tracking
    record_every=99999,
    rng_local=np.random.default_rng(42),
    track_r=True, track_every=200
)

r_track = res_433['r_track']
print(f"  Quench to beta={beta_quench:.2f} (2*beta_c), N={N_433}")
print(f"  Tracked {len(r_track)} <r> measurements")
print()
print(f"  {'Step':>6s}  {'<r>':>8s}")
for step, r_val in r_track:
    print(f"  {step:>6d}  {r_val:>8.4f}")

# Find when minimum is reached
min_idx = np.argmin([r for _, r in r_track])
min_step, min_r_433 = r_track[min_idx]
print(f"\n  MIN <r> = {min_r_433:.4f} at step {min_step}")
print(f"  Initial <r> = {r_track[0][1]:.4f}")
print(f"  Final <r> = {r_track[-1][1]:.4f}")
print(f"  Time: {time.time()-t0:.1f}s")
print()


# ============================================================
# IDEA 434: FULL P(s) DISTRIBUTION AT THE TRANSITION
# ============================================================

print("=" * 78)
print("IDEA 434: FULL P(s) AT THE TRANSITION")
print("Collect spacings at beta=0, beta_c, 2*beta_c, 5*beta_c")
print("=" * 78)
print()

t0 = time.time()
N_434 = 50
EPS_434 = 0.12
bc_434 = beta_c(N_434, EPS_434)

beta_targets_434 = {
    '0': 0.0,
    '1.0*bc': 1.0 * bc_434,
    '1.5*bc': 1.5 * bc_434,
    '2.0*bc': 2.0 * bc_434,
    '3.0*bc': 3.0 * bc_434,
    '5.0*bc': 5.0 * bc_434,
}

for label, beta in beta_targets_434.items():
    all_spacings = []
    for run in range(3):
        res = run_mcmc_with_tracking(
            N_434, beta, EPS_434, n_steps=12000, n_therm=8000,
            record_every=200, rng_local=np.random.default_rng(42 + run*100)
        )
        for cs in res['samples']:
            evals = iDelta_eigenvalues(cs)
            s_normalized = spacing_distribution(evals)
            all_spacings.extend(s_normalized.tolist())

    all_spacings = np.array(all_spacings)
    if len(all_spacings) > 5:
        # Compute histogram (10 bins from 0 to 4)
        hist, bin_edges = np.histogram(all_spacings, bins=10, range=(0, 4), density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Check for bimodality using the dip or kurtosis
        kurtosis = float(stats.kurtosis(all_spacings))
        skewness = float(stats.skew(all_spacings))

        r_val = level_spacing_ratio(np.sort(all_spacings))

        print(f"  beta={label}  ({len(all_spacings)} spacings)")
        print(f"    <s>={np.mean(all_spacings):.4f}, std={np.std(all_spacings):.4f}")
        print(f"    skew={skewness:.4f}, kurtosis={kurtosis:.4f}")
        print(f"    P(s) histogram: ", end="")
        for bc_val, h in zip(bin_centers, hist):
            print(f"{bc_val:.1f}:{h:.3f} ", end="")
        print()

        # Check if it looks like a mixture
        # Bimodal test: does the histogram have a valley?
        diffs = np.diff(hist)
        has_valley = any(diffs[i] < 0 and diffs[j] > 0
                        for i in range(len(diffs))
                        for j in range(i+1, len(diffs)))
        print(f"    Bimodal (valley in P(s))? {'YES' if has_valley else 'NO'}")
    print()

print(f"  Time: {time.time()-t0:.1f}s")
print()


# ============================================================
# IDEA 435: SUB-POISSON <r> vs ACTION BIMODALITY
# ============================================================

print("=" * 78)
print("IDEA 435: IS SUB-POISSON <r> CORRELATED WITH ACTION BIMODALITY?")
print("At each beta, check action histogram for bimodality AND <r>")
print("=" * 78)
print()

t0 = time.time()
N_435 = 50
EPS_435 = 0.12
bc_435 = beta_c(N_435, EPS_435)

beta_mults_435 = np.linspace(0.5, 4.0, 15)

for bm in beta_mults_435:
    beta = bm * bc_435
    all_actions = []
    all_r = []

    for run in range(3):
        res = run_mcmc_with_tracking(
            N_435, beta, EPS_435, n_steps=15000, n_therm=8000,
            record_every=100, rng_local=np.random.default_rng(42 + run*100)
        )
        all_actions.extend(res['actions'].tolist())
        for cs in res['samples']:
            evals = iDelta_eigenvalues(cs)
            all_r.append(level_spacing_ratio(evals))

    actions_arr = np.array(all_actions)
    r_arr = np.array([r for r in all_r if not np.isnan(r)])

    # Check action bimodality via Hartigan's dip test proxy:
    # use kurtosis (bimodal often has negative excess kurtosis)
    action_kurtosis = float(stats.kurtosis(actions_arr)) if len(actions_arr) > 5 else np.nan
    action_std = float(np.std(actions_arr))

    # Simple bimodality check: histogram with 20 bins
    hist_a, bin_edges_a = np.histogram(actions_arr, bins=20)
    # Valley detection
    diffs_a = np.diff(hist_a.astype(float))
    bimodal_action = False
    valley_depth = 0
    for ii in range(len(diffs_a) - 1):
        if diffs_a[ii] < 0:
            for jj in range(ii + 1, len(diffs_a)):
                if diffs_a[jj] > 0:
                    valley_depth = max(valley_depth,
                                      min(hist_a[ii], hist_a[jj+1]) - hist_a[ii+1])
                    bimodal_action = True
                    break

    mean_r = np.mean(r_arr) if len(r_arr) > 0 else np.nan
    print(f"  beta/bc={bm:.2f}  <r>={mean_r:.4f}  "
          f"<S>/N={np.mean(actions_arr)/N_435:.4f}  "
          f"action_kurt={action_kurtosis:.3f}  "
          f"bimodal={bimodal_action}  valley={valley_depth:.0f}")

print(f"\n  Time: {time.time()-t0:.1f}s")
print()


# ============================================================
# IDEA 436: MIXTURE TEST — BLEND CONTINUUM + KR CONFIGS
# ============================================================

print("=" * 78)
print("IDEA 436: MIXTURE TEST — 50/50 CONTINUUM + KR EIGENVALUE BLEND")
print("Does mixing two GUE-like ensembles produce sub-Poisson <r>?")
print("=" * 78)
print()

t0 = time.time()
N_436 = 50
EPS_436 = 0.12
bc_436 = beta_c(N_436, EPS_436)

# Collect continuum configs (beta=0)
print("  Collecting continuum configs (beta=0)...")
continuum_evals = []
for run in range(5):
    res = run_mcmc_with_tracking(
        N_436, 0.0, EPS_436, n_steps=10000, n_therm=5000,
        record_every=200, rng_local=np.random.default_rng(42 + run*10)
    )
    for cs in res['samples']:
        evals = iDelta_eigenvalues(cs)
        continuum_evals.append(evals)

print(f"    Got {len(continuum_evals)} continuum spectra")

# Collect KR configs (beta=5*bc)
print("  Collecting KR configs (beta=5*bc)...")
kr_evals = []
for run in range(5):
    res = run_mcmc_with_tracking(
        N_436, 5.0 * bc_436, EPS_436, n_steps=10000, n_therm=5000,
        record_every=200, rng_local=np.random.default_rng(42 + run*10)
    )
    for cs in res['samples']:
        evals = iDelta_eigenvalues(cs)
        kr_evals.append(evals)

print(f"    Got {len(kr_evals)} KR spectra")

# Pure <r> for each ensemble
cont_r = [level_spacing_ratio(e) for e in continuum_evals]
kr_r = [level_spacing_ratio(e) for e in kr_evals]
cont_r_clean = [r for r in cont_r if not np.isnan(r)]
kr_r_clean = [r for r in kr_r if not np.isnan(r)]

print(f"  Continuum <r> = {np.mean(cont_r_clean):.4f} +/- {np.std(cont_r_clean):.4f}")
print(f"  KR <r> = {np.mean(kr_r_clean):.4f} +/- {np.std(kr_r_clean):.4f}")

# Mixture: randomly pair one continuum + one KR spectrum,
# concatenate eigenvalues, compute <r>
n_mix = min(len(continuum_evals), len(kr_evals))
mix_r_values = []
for i in range(n_mix):
    mixed = np.concatenate([continuum_evals[i], kr_evals[i]])
    mix_r_values.append(level_spacing_ratio(mixed))

mix_r_clean = [r for r in mix_r_values if not np.isnan(r)]
print(f"  MIXED (concat eigenvalues) <r> = {np.mean(mix_r_clean):.4f} +/- {np.std(mix_r_clean):.4f}")

# Also try: interleave samples (alternating continuum and KR eigenvalues)
# This simulates a chain that flips between phases
mix2_r = []
for i in range(min(n_mix - 1, 50)):
    # Take first half of evals from continuum, second half from KR
    half = len(continuum_evals[i]) // 2
    mixed2 = np.sort(np.concatenate([continuum_evals[i][:half], kr_evals[i][:half]]))
    mix2_r.append(level_spacing_ratio(mixed2))

mix2_clean = [r for r in mix2_r if not np.isnan(r)]
if len(mix2_clean) > 0:
    print(f"  MIXED (half/half sorted) <r> = {np.mean(mix2_clean):.4f} +/- {np.std(mix2_clean):.4f}")

# Key test: does the mixture <r> match the transition <r>?
print(f"\n  KEY: If mixture <r> << 0.386 (Poisson), the phase coexistence explanation is supported")
print(f"  Time: {time.time()-t0:.1f}s")
print()


# ============================================================
# IDEA 437: LEVEL SPACING RATIO OF WIGHTMAN FUNCTION W
# ============================================================

print("=" * 78)
print("IDEA 437: <r> OF WIGHTMAN FUNCTION W (not just iDelta)")
print("=" * 78)
print()

t0 = time.time()
N_437 = 50
EPS_437 = 0.12
bc_437 = beta_c(N_437, EPS_437)

beta_targets_437 = {
    '0': 0.0,
    '1.0*bc': 1.0 * bc_437,
    '2.0*bc': 2.0 * bc_437,
    '5.0*bc': 5.0 * bc_437,
}

for label, beta in beta_targets_437.items():
    r_iDelta_vals = []
    r_W_vals = []

    for run in range(3):
        res = run_mcmc_with_tracking(
            N_437, beta, EPS_437, n_steps=12000, n_therm=8000,
            record_every=400, rng_local=np.random.default_rng(42 + run*100)
        )
        for cs in res['samples']:
            # iDelta eigenvalues
            evals_iD = iDelta_eigenvalues(cs)
            r_iDelta_vals.append(level_spacing_ratio(evals_iD))

            # Wightman eigenvalues
            W = sj_wightman_function(cs)
            evals_W = np.linalg.eigvalsh(W)
            r_W_vals.append(level_spacing_ratio(evals_W))

    r_iD = np.array([r for r in r_iDelta_vals if not np.isnan(r)])
    r_W = np.array([r for r in r_W_vals if not np.isnan(r)])

    print(f"  beta={label}:")
    if len(r_iD) > 0:
        print(f"    iDelta <r> = {np.mean(r_iD):.4f} +/- {np.std(r_iD):.4f}")
    if len(r_W) > 0:
        print(f"    W      <r> = {np.mean(r_W):.4f} +/- {np.std(r_W):.4f}")
    print()

print(f"  Time: {time.time()-t0:.1f}s")
print()


# ============================================================
# IDEA 438: <r> AT THE 4D BD TRANSITION
# ============================================================

print("=" * 78)
print("IDEA 438: <r> AT THE 4D BD TRANSITION")
print("Sprinkle into 4D Minkowski diamond, compute BD4 action, scan beta")
print("=" * 78)
print()

t0 = time.time()
N_438 = 30  # smaller for 4D (expensive)

# For 4D, beta_c is different. We'll scan empirically.
# Use sprinkled causets with a simple reweighting.
# Actually: the 4D transition is in MCMC over sprinkled causets, which
# we don't have for 4D (no 4-orders). Instead, test: does the 2-order
# transition at DIFFERENT N produce the same phenomenon?
# OR: sprinkle into 4D, compute iDelta eigenvalues, apply BD4 action weighting.

# Approach: Sprinkle 4D causets, compute iDelta <r> for unweighted ensemble.
# Then compare with 2D sprinkled at same N.

print("  4D sprinkled causet (no MCMC, just sprinkled ensemble):")
r_4d_vals = []
for trial in range(20):
    cs_4d, coords_4d = sprinkle_fast(N_438, dim=4, extent_t=1.0, rng=rng)
    if cs_4d.num_relations() > 2:
        evals = iDelta_eigenvalues(cs_4d)
        r_4d_vals.append(level_spacing_ratio(evals))

r_4d_clean = [r for r in r_4d_vals if not np.isnan(r)]
if len(r_4d_clean) > 0:
    print(f"    4D sprinkled <r> = {np.mean(r_4d_clean):.4f} +/- {np.std(r_4d_clean):.4f} ({len(r_4d_clean)} trials)")

print("  2D sprinkled causet (for comparison):")
r_2d_vals = []
for trial in range(20):
    cs_2d, coords_2d = sprinkle_fast(N_438, dim=2, extent_t=1.0, rng=rng)
    if cs_2d.num_relations() > 2:
        evals = iDelta_eigenvalues(cs_2d)
        r_2d_vals.append(level_spacing_ratio(evals))

r_2d_clean = [r for r in r_2d_vals if not np.isnan(r)]
if len(r_2d_clean) > 0:
    print(f"    2D sprinkled <r> = {np.mean(r_2d_clean):.4f} +/- {np.std(r_2d_clean):.4f} ({len(r_2d_clean)} trials)")

# Now: 2-order MCMC in pseudo-4D (use 2-orders but with 4D BD action as weight)
# This is not standard but shows if the dip is action-dependent
print("\n  2-order MCMC with 4D BD action weighting:")

beta_mults_438 = [0.0, 0.5, 1.0, 2.0, 5.0]
for bm in beta_mults_438:
    # Use a different beta_c estimate for 4D action
    # BD4 action is much smaller than BD2, so we need higher beta
    beta = bm * 10.0  # empirical scale

    current = TwoOrder(N_438, rng=rng)
    current_cs = current.to_causet()
    current_S = bd_action_4d(current_cs)
    n_acc = 0
    r_vals_438 = []

    for step in range(8000):
        proposed = swap_move(current, rng)
        proposed_cs = proposed.to_causet()
        proposed_S = bd_action_4d(proposed_cs)

        dS = beta * (proposed_S - current_S)
        if dS <= 0 or rng.random() < np.exp(-dS):
            current = proposed
            current_cs = proposed_cs
            current_S = proposed_S
            n_acc += 1

        if step >= 5000 and (step - 5000) % 300 == 0:
            evals = iDelta_eigenvalues(current_cs)
            r_vals_438.append(level_spacing_ratio(evals))

    r_clean = [r for r in r_vals_438 if not np.isnan(r)]
    if len(r_clean) > 0:
        print(f"    beta_4D={beta:.1f} (bm={bm:.1f})  <r>={np.mean(r_clean):.4f} +/- {np.std(r_clean):.4f}  "
              f"accept={n_acc/8000:.3f}")

print(f"\n  Time: {time.time()-t0:.1f}s")
print()


# ============================================================
# IDEA 439: EPSILON DEPENDENCE — SHARPER TRANSITION → DEEPER DIP?
# ============================================================

print("=" * 78)
print("IDEA 439: EPSILON DEPENDENCE OF THE SUB-POISSON DIP")
print("eps=0.05 (sharp), 0.12 (standard), 0.25 (soft)")
print("=" * 78)
print()

t0 = time.time()
N_439 = 50
eps_values = [0.05, 0.12, 0.25]

for eps_val in eps_values:
    bc_val = beta_c(N_439, eps_val)
    beta_mults_439 = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    print(f"  eps={eps_val}, beta_c={bc_val:.2f}:")

    for bm in beta_mults_439:
        beta = bm * bc_val
        r_values_439 = []

        for run in range(2):
            res = run_mcmc_with_tracking(
                N_439, beta, eps_val, n_steps=12000, n_therm=8000,
                record_every=200, rng_local=np.random.default_rng(42 + run*100)
            )
            for cs in res['samples']:
                evals = iDelta_eigenvalues(cs)
                r_values_439.append(level_spacing_ratio(evals))

        r_clean = [r for r in r_values_439 if not np.isnan(r)]
        mean_r = np.mean(r_clean) if len(r_clean) > 0 else np.nan
        std_r = np.std(r_clean) if len(r_clean) > 0 else np.nan
        print(f"    beta/bc={bm:.1f}  <r>={mean_r:.4f} +/- {std_r:.4f}")

    print()

print(f"  Time: {time.time()-t0:.1f}s")
print()


# ============================================================
# IDEA 440: PARTIALLY ORDERED TOTAL ORDER — <r> FOR A DEGRADED CHAIN
# ============================================================

print("=" * 78)
print("IDEA 440: <r> FOR PARTIALLY ORDERED TOTAL ORDER")
print("Start from chain (signum matrix, <r>->1), randomly remove relations")
print("=" * 78)
print()

t0 = time.time()
N_440 = 50

# Build a total order (chain): order[i,j] = True iff i < j
def make_chain(N):
    cs = FastCausalSet(N)
    for i in range(N):
        for j in range(i+1, N):
            cs.order[i, j] = True
    return cs

# Remove a fraction of non-link relations (keep transitivity-required ones)
def degrade_chain(N, removal_frac, rng_local):
    """Start from chain. Remove `removal_frac` of relations randomly.
    After removal, take transitive closure of what remains."""
    cs = make_chain(N)

    # Get all relations (upper triangle)
    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            pairs.append((i, j))

    # Randomly select which to remove
    n_remove = int(len(pairs) * removal_frac)
    remove_idx = rng_local.choice(len(pairs), size=n_remove, replace=False)

    for idx in remove_idx:
        i, j = pairs[idx]
        cs.order[i, j] = False

    # Transitive closure (ensure consistency)
    # Floyd-Warshall style
    changed = True
    while changed:
        old_order = cs.order.copy()
        # If i<k and k<j then i<j
        cs.order = cs.order | (cs.order.astype(np.int32) @ cs.order.astype(np.int32) > 0)
        # But we can't ADD relations this way if we removed them
        # Actually the question is: remove relations, then DON'T restore transitivity.
        # A partial order is still valid if we just remove relations (the result
        # is still a partial order as long as antisymmetry holds).
        break

    # Actually: just removing relations from a total order gives a valid partial order
    # (subset of a partial order is a partial order)
    # So just remove and don't re-close. But we need to remove the transitive
    # closure too if we remove a generating link.
    # Simplest: just remove random relations.
    return cs


# Alternative cleaner approach: start from chain, remove links only
def degrade_chain_links(N, removal_frac, rng_local):
    """Start from chain. Remove `removal_frac` of LINKS (nearest-neighbor relations)
    and recompute transitive closure."""
    # Chain links are (i, i+1) for all i
    links = list(range(N-1))  # link i means i -> i+1
    n_remove = int(len(links) * removal_frac)
    remove = set(rng_local.choice(links, size=n_remove, replace=False))

    # Remaining links
    remaining = [l for l in links if l not in remove]

    # Build connected segments and take transitive closure
    cs = FastCausalSet(N)
    for l in remaining:
        cs.order[l, l+1] = True

    # Transitive closure
    for _ in range(N):
        new_order = cs.order | (cs.order.astype(np.int32) @ cs.order.astype(np.int32) > 0)
        if np.array_equal(new_order, cs.order):
            break
        cs.order = new_order

    return cs


# Full chain
cs_chain = make_chain(N_440)
evals_chain = iDelta_eigenvalues(cs_chain)
r_chain = level_spacing_ratio(evals_chain)
print(f"  Full chain (total order): <r> = {r_chain:.4f}, relations = {cs_chain.num_relations()}")

# Scan removal fractions
removal_fracs = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]

print(f"\n  Method 1: Remove random relations (keeps partial order)")
for frac in removal_fracs:
    r_vals_440 = []
    for trial in range(5):
        cs_degraded = degrade_chain(N_440, frac, np.random.default_rng(42 + trial))
        evals = iDelta_eigenvalues(cs_degraded)
        r_vals_440.append(level_spacing_ratio(evals))

    r_clean = [r for r in r_vals_440 if not np.isnan(r)]
    if len(r_clean) > 0:
        rels = degrade_chain(N_440, frac, np.random.default_rng(42)).num_relations()
        print(f"    removal={frac:.2f}  <r>={np.mean(r_clean):.4f} +/- {np.std(r_clean):.4f}  "
              f"relations={rels}")

print(f"\n  Method 2: Remove links and re-close")
for frac in removal_fracs:
    r_vals_440b = []
    for trial in range(5):
        cs_degraded2 = degrade_chain_links(N_440, frac, np.random.default_rng(42 + trial))
        evals = iDelta_eigenvalues(cs_degraded2)
        r_vals_440b.append(level_spacing_ratio(evals))

    r_clean = [r for r in r_vals_440b if not np.isnan(r)]
    if len(r_clean) > 0:
        rels = degrade_chain_links(N_440, frac, np.random.default_rng(42)).num_relations()
        print(f"    link_removal={frac:.2f}  <r>={np.mean(r_clean):.4f} +/- {np.std(r_clean):.4f}  "
              f"relations={rels}")

print(f"\n  Time: {time.time()-t0:.1f}s")

print()
print("=" * 78)
print("EXPERIMENT 92 COMPLETE")
print("=" * 78)

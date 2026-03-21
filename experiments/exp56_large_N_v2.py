"""
Experiment 56: Large-N v2 — Three Targeted Experiments

Experiment A: SJ Vacuum on SPRINKLED Causets (not 2-orders)
  Sprinkle N points into a 2D causal diamond in Minkowski space.
  Key question: Does c_eff converge to c=1? (On 2-orders it diverges 3→4.1)

Experiment B: GUE at the BD Phase Transition
  Scan beta from 0 to 3*beta_c. Does <r> jump (first-order) or change smoothly?

Experiment C: Spectral Compressibility — Deep GUE Characterization
  Compute number variance Sigma^2(L). GUE: ~(1/pi^2)ln(L). Poisson: L.
  What are causets?
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats
from causal_sets.fast_core import FastCausalSet
from causal_sets.two_orders import TwoOrder, swap_move
from causal_sets.two_orders_v2 import bd_action_corrected

rng = np.random.default_rng(42)


# ============================================================
# SHARED UTILITIES
# ============================================================

def sj_vacuum(cs):
    """Compute SJ Wightman function W and eigenvalues of i*Delta."""
    N = cs.n
    C = cs.order.astype(float)
    Delta = (2.0 / N) * (C.T - C)
    H = 1j * Delta
    evals, evecs = np.linalg.eigh(H)
    evals = evals.real
    pos = evals > 1e-12
    if not np.any(pos):
        return np.zeros((N, N)), evals
    W = (evecs[:, pos] @ np.diag(evals[pos]) @ evecs[:, pos].conj().T).real
    w_max = np.linalg.eigvalsh(W).max()
    if w_max > 1:
        W = W / w_max
    return W, evals


def entanglement_entropy(W, region):
    """Von Neumann entropy of reduced state on region."""
    if len(region) < 2:
        return 0.0
    W_A = W[np.ix_(region, region)]
    eigs = np.linalg.eigvalsh(W_A)
    eigs = np.clip(eigs, 1e-15, 1 - 1e-15)
    return float(-np.sum(eigs * np.log(eigs) + (1 - eigs) * np.log(1 - eigs)))


def level_spacing_ratio(eigenvalues):
    """Compute mean level spacing ratio <r> (GUE ~ 0.603, Poisson ~ 0.386)."""
    evals = np.sort(eigenvalues)
    # Use only positive eigenvalues (the SJ spectrum is ±symmetric)
    pos = evals[evals > 1e-12]
    if len(pos) < 4:
        return float('nan')
    spacings = np.diff(pos)
    spacings = spacings[spacings > 1e-15]
    if len(spacings) < 3:
        return float('nan')
    r_vals = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
    return float(np.mean(r_vals))


def number_variance(eigenvalues, L_values):
    """
    Compute number variance Sigma^2(L) for the unfolded spectrum.

    Sigma^2(L) = Var(number of eigenvalues in a window of width L mean spacings).

    Uses the positive eigenvalues, unfolded to unit mean spacing.
    """
    evals = np.sort(eigenvalues)
    pos = evals[evals > 1e-12]
    if len(pos) < 10:
        return [float('nan')] * len(L_values)

    # Unfold: map to uniform density (unit mean spacing)
    # Simple unfolding: rank / N_pos gives the integrated density
    N_pos = len(pos)
    unfolded = np.arange(N_pos, dtype=float)  # Already uniformly spaced in rank
    # Actually: unfolded[i] = i, so mean spacing = 1
    # Better: use actual eigenvalues, rescaled so mean spacing = 1
    mean_spacing = np.mean(np.diff(pos))
    if mean_spacing < 1e-15:
        return [float('nan')] * len(L_values)
    unfolded = pos / mean_spacing

    results = []
    for L in L_values:
        # Slide a window of width L across the spectrum
        counts = []
        # Window center ranges from min to max - L
        e_min = unfolded[0]
        e_max = unfolded[-1]
        if e_max - e_min < L:
            results.append(float('nan'))
            continue

        # Use many windows
        n_windows = min(200, int((e_max - e_min - L) / (L * 0.1)) + 1)
        centers = np.linspace(e_min + L/2, e_max - L/2, n_windows)
        for c in centers:
            n_in = np.sum((unfolded >= c - L/2) & (unfolded < c + L/2))
            counts.append(n_in)

        counts = np.array(counts, dtype=float)
        results.append(float(np.var(counts)))

    return results


def sprinkle_causal_diamond_2d(N, rng):
    """
    Sprinkle N points uniformly into a 2D causal diamond in Minkowski space.

    The diamond is {(t,x) : 0 < t-x < 1, 0 < t+x < 1}, i.e., the intersection
    of the future lightcone of (0,0) with the past lightcone of (1,0).

    In lightcone coordinates u = t+x, v = t-x: the diamond is [0,1] x [0,1].
    So we sprinkle u, v ~ Uniform(0,1), then t = (u+v)/2, x = (u-v)/2.

    Causal relation: i < j iff u_i < u_j AND v_i < v_j (both lightcone coords increase).
    """
    # Sprinkle in lightcone coordinates
    u = rng.uniform(0, 1, N)
    v = rng.uniform(0, 1, N)

    # Convert to (t, x)
    t = (u + v) / 2
    x = (u - v) / 2

    # Sort by time
    order = np.argsort(t)
    t = t[order]
    x = x[order]
    u = u[order]
    v = v[order]

    # Build causal order: i < j iff u_i < u_j AND v_i < v_j
    # This is equivalent to |dt| > |dx| and dt > 0 (lightcone causality)
    cs = FastCausalSet(N)
    u_less = u[:, None] < u[None, :]
    v_less = v[:, None] < v[None, :]
    cs.order = u_less & v_less

    return cs, np.column_stack([t, x])


def random_dag_matched(N, target_density, rng):
    """Generate a random DAG with approximately matched ordering fraction."""
    cs = FastCausalSet(N)
    # For a random upper triangular matrix with probability p,
    # after transitive closure the density is higher. Start with lower p.
    p = target_density * 0.5  # heuristic
    mask = rng.random((N, N)) < p
    cs.order = np.triu(mask, k=1)
    # Transitive closure
    order_int = cs.order.astype(np.int32)
    for _ in range(int(np.log2(N)) + 1):
        new_order = (order_int @ order_int > 0) | cs.order
        if np.array_equal(new_order, cs.order):
            break
        cs.order = new_order
        order_int = cs.order.astype(np.int32)
    return cs


# ============================================================
# EXPERIMENT A: SJ VACUUM ON SPRINKLED CAUSETS
# ============================================================
print("=" * 78)
print("EXPERIMENT A: SJ VACUUM ON SPRINKLED CAUSETS (NOT 2-ORDERS)")
print("Key question: Does c_eff converge to c=1 on sprinkled causets?")
print("On 2-orders, c_eff diverges (3 → 4.1 from N=50 → 500). BAD.")
print("Sprinkled causets have ACTUAL Minkowski embedding. Should be better.")
print("=" * 78)

N_sizes_A = [50, 100, 200, 300]
n_trials_A = 3

print("\n--- Part 1: c_eff on sprinkled causets ---")
print(f"  c_eff = 3 * S(N/2) / ln(N)")
print(f"  CFT prediction for free scalar: c = 1\n")

results_A = {}

for N in N_sizes_A:
    c_effs_sprinkle = []
    c_effs_2order = []
    c_effs_null = []
    r_vals_sprinkle = []
    r_vals_2order = []
    r_vals_null = []
    densities_sprinkle = []

    for trial in range(n_trials_A):
        print(f"  N={N}, trial {trial+1}/{n_trials_A}...", flush=True)

        # --- Sprinkled causet ---
        cs_spr, coords = sprinkle_causal_diamond_2d(N, rng)
        density_spr = cs_spr.ordering_fraction()
        densities_sprinkle.append(density_spr)

        W_spr, evals_spr = sj_vacuum(cs_spr)
        region_half = list(range(N // 2))
        S_spr = entanglement_entropy(W_spr, region_half)
        c_eff_spr = 3 * S_spr / np.log(N)
        c_effs_sprinkle.append(c_eff_spr)
        r_spr = level_spacing_ratio(evals_spr)
        r_vals_sprinkle.append(r_spr)

        # --- 2-order (for comparison) ---
        to = TwoOrder(N, rng=rng)
        cs_2o = to.to_causet()
        W_2o, evals_2o = sj_vacuum(cs_2o)
        S_2o = entanglement_entropy(W_2o, region_half)
        c_eff_2o = 3 * S_2o / np.log(N)
        c_effs_2order.append(c_eff_2o)
        r_2o = level_spacing_ratio(evals_2o)
        r_vals_2order.append(r_2o)

        # --- Null: random DAG with matched density ---
        cs_null = random_dag_matched(N, density_spr, rng)
        W_null, evals_null = sj_vacuum(cs_null)
        S_null = entanglement_entropy(W_null, region_half)
        c_eff_null = 3 * S_null / np.log(N)
        c_effs_null.append(c_eff_null)
        r_null = level_spacing_ratio(evals_null)
        r_vals_null.append(r_null)

    results_A[N] = {
        'c_eff_sprinkle': c_effs_sprinkle,
        'c_eff_2order': c_effs_2order,
        'c_eff_null': c_effs_null,
        'r_sprinkle': r_vals_sprinkle,
        'r_2order': r_vals_2order,
        'r_null': r_vals_null,
        'density_sprinkle': densities_sprinkle,
    }

    print(f"    Sprinkled: c_eff = {np.mean(c_effs_sprinkle):.3f} +/- {np.std(c_effs_sprinkle):.3f}, "
          f"<r> = {np.nanmean(r_vals_sprinkle):.3f}, density = {np.mean(densities_sprinkle):.3f}")
    print(f"    2-order:   c_eff = {np.mean(c_effs_2order):.3f} +/- {np.std(c_effs_2order):.3f}, "
          f"<r> = {np.nanmean(r_vals_2order):.3f}")
    print(f"    Null DAG:  c_eff = {np.mean(c_effs_null):.3f} +/- {np.std(c_effs_null):.3f}, "
          f"<r> = {np.nanmean(r_vals_null):.3f}")

# Summary table
print("\n--- SUMMARY TABLE: c_eff convergence ---")
print(f"  {'N':>5}  {'Sprinkled c_eff':>15}  {'2-order c_eff':>15}  {'Null c_eff':>12}  {'Sprinkled <r>':>14}")
print(f"  {'='*5}  {'='*15}  {'='*15}  {'='*12}  {'='*14}")
for N in N_sizes_A:
    r = results_A[N]
    print(f"  {N:>5}  {np.mean(r['c_eff_sprinkle']):>8.3f}+/-{np.std(r['c_eff_sprinkle']):.3f}"
          f"  {np.mean(r['c_eff_2order']):>8.3f}+/-{np.std(r['c_eff_2order']):.3f}"
          f"  {np.mean(r['c_eff_null']):>8.3f}+/-{np.std(r['c_eff_null']):.3f}"
          f"  {np.nanmean(r['r_sprinkle']):>8.3f}")

# Check: is sprinkled c_eff converging to 1?
sprinkled_c_effs = [np.mean(results_A[N]['c_eff_sprinkle']) for N in N_sizes_A]
print(f"\n  Sprinkled c_eff trend: {' -> '.join(f'{c:.2f}' for c in sprinkled_c_effs)}")
if len(sprinkled_c_effs) >= 2:
    if sprinkled_c_effs[-1] < sprinkled_c_effs[0]:
        print(f"  DECREASING — potentially converging toward c=1")
    else:
        print(f"  NOT DECREASING — diverging like 2-orders")


# ============================================================
# EXPERIMENT B: GUE AT THE BD PHASE TRANSITION
# ============================================================
print("\n\n" + "=" * 78)
print("EXPERIMENT B: GUE AT THE BD PHASE TRANSITION")
print("Does <r> jump discontinuously (first-order) at beta_c?")
print("Or does it change continuously (second-order)?")
print("=" * 78)

N_B = 50
eps_B = 0.12
beta_c = 1.66 / (N_B * eps_B**2)
print(f"\n  N = {N_B}, eps = {eps_B}, beta_c = {beta_c:.2f}")

beta_fracs = [0.0, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0]
betas = [f * beta_c for f in beta_fracs]

n_mcmc_steps = 20000
n_therm = 10000
n_samples_per_beta = 5  # take multiple samples after thermalization

print(f"\n--- Scanning beta from 0 to {3*beta_c:.1f} ({len(betas)} points) ---")
print(f"  MCMC: {n_mcmc_steps} steps, {n_therm} thermalization\n")

results_B = {}

for i, beta in enumerate(betas):
    frac = beta_fracs[i]
    print(f"  beta = {frac:.1f} * beta_c = {beta:.2f}...", end=" ", flush=True)

    # Run MCMC
    current = TwoOrder(N_B, rng=rng)
    current_cs = current.to_causet()
    current_S = bd_action_corrected(current_cs, eps_B)

    n_acc = 0
    r_values = []
    ordering_fracs = []
    actions_recorded = []

    record_every = max(1, (n_mcmc_steps - n_therm) // n_samples_per_beta)

    for step in range(n_mcmc_steps):
        proposed = swap_move(current, rng)
        proposed_cs = proposed.to_causet()
        proposed_S = bd_action_corrected(proposed_cs, eps_B)

        dS = beta * (proposed_S - current_S)
        if dS <= 0 or (beta > 0 and rng.random() < np.exp(-min(dS, 500))):
            current = proposed
            current_cs = proposed_cs
            current_S = proposed_S
            n_acc += 1

        if step >= n_therm and (step - n_therm) % record_every == 0:
            # Compute observables on this sample
            W, evals = sj_vacuum(current_cs)
            r = level_spacing_ratio(evals)
            r_values.append(r)
            ordering_fracs.append(current_cs.ordering_fraction())
            actions_recorded.append(current_S)

    accept_rate = n_acc / n_mcmc_steps
    mean_r = np.nanmean(r_values) if r_values else float('nan')
    mean_of = np.mean(ordering_fracs) if ordering_fracs else float('nan')
    mean_S = np.mean(actions_recorded) if actions_recorded else float('nan')

    results_B[frac] = {
        'beta': beta,
        'r_values': r_values,
        'mean_r': mean_r,
        'ordering_frac': mean_of,
        'mean_action': mean_S,
        'accept_rate': accept_rate,
    }

    print(f"<r> = {mean_r:.3f}, OF = {mean_of:.3f}, S/N = {mean_S/N_B:.3f}, "
          f"accept = {accept_rate:.2f}")

print("\n--- SUMMARY: <r> vs beta/beta_c ---")
print(f"  {'beta/beta_c':>11}  {'<r>':>8}  {'OF':>8}  {'S/N':>8}  Notes")
print(f"  {'='*11}  {'='*8}  {'='*8}  {'='*8}  {'='*25}")
for frac in beta_fracs:
    r = results_B[frac]
    notes = ""
    if abs(r['mean_r'] - 0.603) < 0.03:
        notes = "GUE"
    elif abs(r['mean_r'] - 0.386) < 0.03:
        notes = "Poisson"
    elif r['mean_r'] > 0.55:
        notes = "near-GUE"
    elif r['mean_r'] < 0.42:
        notes = "near-Poisson"
    else:
        notes = "intermediate"
    print(f"  {frac:>11.1f}  {r['mean_r']:>8.3f}  {r['ordering_frac']:>8.3f}  "
          f"{r['mean_action']/N_B:>8.3f}  {notes}")

# Check: is the transition sharp or smooth?
r_below = [results_B[f]['mean_r'] for f in beta_fracs if f < 1.0]
r_above = [results_B[f]['mean_r'] for f in beta_fracs if f > 1.0]
if r_below and r_above:
    mean_below = np.nanmean(r_below)
    mean_above = np.nanmean(r_above)
    print(f"\n  Mean <r> below beta_c: {mean_below:.3f}")
    print(f"  Mean <r> above beta_c: {mean_above:.3f}")
    print(f"  Jump magnitude: {abs(mean_above - mean_below):.3f}")
    if abs(mean_above - mean_below) > 0.1:
        print(f"  DISCONTINUOUS JUMP — suggests first-order transition")
    else:
        print(f"  SMOOTH CROSSOVER — no sharp transition in <r>")


# ============================================================
# EXPERIMENT C: SPECTRAL COMPRESSIBILITY — DEEP GUE CHARACTERIZATION
# ============================================================
print("\n\n" + "=" * 78)
print("EXPERIMENT C: SPECTRAL COMPRESSIBILITY")
print("Number variance Sigma^2(L): GUE ~ (1/pi^2)*ln(L), Poisson ~ L")
print("What are causets? This determines if the GUE signature is local or global.")
print("=" * 78)

N_sizes_C = [50, 100, 200, 300]
L_values = [1, 2, 4, 8, 16]
n_trials_C = 3

print(f"\n--- Sigma^2(L) for L = {L_values} ---\n")

results_C = {}

for N in N_sizes_C:
    sigma2_all = {L: [] for L in L_values}
    sigma2_null = {L: [] for L in L_values}

    for trial in range(n_trials_C):
        print(f"  N={N}, trial {trial+1}/{n_trials_C}...", flush=True)

        # 2-order causet
        to = TwoOrder(N, rng=rng)
        cs = to.to_causet()
        W, evals = sj_vacuum(cs)

        nv = number_variance(evals, L_values)
        for i, L in enumerate(L_values):
            sigma2_all[L].append(nv[i])

        # Null: random DAG
        density = cs.ordering_fraction()
        cs_null = random_dag_matched(N, density, rng)
        W_null, evals_null = sj_vacuum(cs_null)
        nv_null = number_variance(evals_null, L_values)
        for i, L in enumerate(L_values):
            sigma2_null[L].append(nv_null[i])

    results_C[N] = {
        'sigma2': {L: np.nanmean(sigma2_all[L]) for L in L_values},
        'sigma2_null': {L: np.nanmean(sigma2_null[L]) for L in L_values},
    }

    print(f"    N={N} Sigma^2(L):")
    for L in L_values:
        gue_pred = (1.0 / np.pi**2) * np.log(L) if L > 1 else 0
        poisson_pred = L
        print(f"      L={L:>2}: causet={np.nanmean(sigma2_all[L]):>7.3f}, "
              f"null={np.nanmean(sigma2_null[L]):>7.3f}, "
              f"GUE={gue_pred:>6.3f}, Poisson={poisson_pred:>3}")

# Fit Sigma^2(L) = a + b*L + c*ln(L) for each N
print("\n--- Fitting Sigma^2(L) = a + b*L + c*ln(L) ---")
for N in N_sizes_C:
    s2 = results_C[N]['sigma2']
    s2_null = results_C[N]['sigma2_null']

    L_arr = np.array(L_values, dtype=float)
    y_causet = np.array([s2[L] for L in L_values])
    y_null = np.array([s2_null[L] for L in L_values])

    # Skip NaNs
    valid_c = ~np.isnan(y_causet)
    valid_n = ~np.isnan(y_null)

    if np.sum(valid_c) >= 3:
        # Design matrix: [1, L, ln(L)]
        X = np.column_stack([np.ones(len(L_arr)), L_arr, np.log(L_arr + 1e-10)])
        Xc = X[valid_c]
        yc = y_causet[valid_c]
        try:
            coeffs_c = np.linalg.lstsq(Xc, yc, rcond=None)[0]
            print(f"\n  N={N} Causet: Sigma^2 = {coeffs_c[0]:.3f} + {coeffs_c[1]:.4f}*L + {coeffs_c[2]:.3f}*ln(L)")
            print(f"    b (linear coeff): {coeffs_c[1]:.4f} (Poisson: 1.0, GUE: 0.0)")
            print(f"    c (log coeff): {coeffs_c[2]:.3f} (GUE: {1/np.pi**2:.3f} = 1/pi^2)")

            # Spectral compressibility: chi = lim Sigma^2(L)/L
            chi = coeffs_c[1]
            print(f"    Spectral compressibility chi = {chi:.4f} (GUE: 0, Poisson: 1)")
        except Exception as e:
            print(f"  N={N} Causet fit failed: {e}")

    if np.sum(valid_n) >= 3:
        Xn = X[valid_n]
        yn = y_null[valid_n]
        try:
            coeffs_n = np.linalg.lstsq(Xn, yn, rcond=None)[0]
            print(f"  N={N} Null:   Sigma^2 = {coeffs_n[0]:.3f} + {coeffs_n[1]:.4f}*L + {coeffs_n[2]:.3f}*ln(L)")
            print(f"    chi_null = {coeffs_n[1]:.4f}")
        except Exception as e:
            print(f"  N={N} Null fit failed: {e}")


# Also measure on sprinkled causets for comparison
print("\n\n--- Sigma^2(L) on SPRINKLED causets (Experiment A+C hybrid) ---")
for N in [50, 100, 200]:
    sigma2_spr = {L: [] for L in L_values}
    for trial in range(n_trials_C):
        cs_spr, _ = sprinkle_causal_diamond_2d(N, rng)
        _, evals_spr = sj_vacuum(cs_spr)
        nv_spr = number_variance(evals_spr, L_values)
        for i, L in enumerate(L_values):
            sigma2_spr[L].append(nv_spr[i])

    print(f"  N={N} Sprinkled Sigma^2(L):")
    for L in L_values:
        gue_pred = (1.0 / np.pi**2) * np.log(L) if L > 1 else 0
        print(f"    L={L:>2}: {np.nanmean(sigma2_spr[L]):>7.3f} (GUE: {gue_pred:.3f})")


# ============================================================
# FINAL SCORING
# ============================================================
print("\n\n" + "=" * 78)
print("FINAL SCORING")
print("=" * 78)

print("""
Scoring criteria:
  - Novelty: Has this been done before? (0-3)
  - Rigor: Clean signal, survives null test? (0-3)
  - Depth: Connection to fundamental physics? (0-2)
  - Audience: Who cares? (0-2)
  Total out of 10.
""")

# Print summary of key numbers for scoring
print("KEY NUMBERS:")
print(f"  Experiment A (Sprinkled c_eff):")
for N in N_sizes_A:
    r = results_A[N]
    print(f"    N={N}: c_eff(sprinkled)={np.mean(r['c_eff_sprinkle']):.2f}, "
          f"c_eff(2-order)={np.mean(r['c_eff_2order']):.2f}, "
          f"<r>(sprinkled)={np.nanmean(r['r_sprinkle']):.3f}")

print(f"\n  Experiment B (Phase transition):")
for frac in beta_fracs:
    r = results_B[frac]
    print(f"    beta/beta_c={frac:.1f}: <r>={r['mean_r']:.3f}")

print(f"\n  Experiment C (Spectral compressibility):")
for N in N_sizes_C:
    s2 = results_C[N]['sigma2']
    print(f"    N={N}: Sigma^2(16)={s2.get(16, float('nan')):.3f}")

print("\nScoring will be done after reviewing all output above.")
print("=" * 78)

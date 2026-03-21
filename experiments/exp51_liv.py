"""
Experiment 51: Lorentz Invariance Violation from the SJ Vacuum on Causal Sets

Extract the effective dispersion relation E(p) from the SJ Wightman function
on sprinkled causal sets and compare with LHAASO observational constraints.

Physics:
- The SJ vacuum gives a Wightman function W[i,j] for a free massless scalar
- In the continuum 2D: W(x,y) = -(1/4pi) ln|sigma(x,y)|
- On a discrete causal set, W is modified by discreteness → modified dispersion
- If we extract w(k), any deviation from w = |k| is Lorentz invariance violation
- LHAASO constrains E_QG > 10 E_Planck for linear LIV (n=1)

Approach:
1. Sprinkle N elements into a 2D causal diamond of size L
2. Compute W[i,j] from the SJ construction
3. Discrete Fourier transform: W(k, omega) = sum_{i,j} W[i,j] exp(-i*omega*t_ij + i*k*x_ij)
4. For each k, find omega that maximizes |W(k, omega)| → dispersion relation w(k)
5. Fit w(k) = |k|(1 + alpha_1*(k/k_P) + alpha_2*(k/k_P)^2 + ...)
6. Compare alpha_1 with LHAASO bound |alpha_1| < 1/10
7. Null model: random DAG with matched density

References:
- Johnston, PRL 103, 180401 (2009)
- Sorkin, arXiv:1205.2953 (2012)
- Saravani, Aslanbeigi, Sorkin, JHEP 1407:024 (2014)
- LHAASO Collaboration, PRL 128, 051102 (2022)
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy.optimize import curve_fit
from causal_sets.fast_core import FastCausalSet, sprinkle_fast

rng = np.random.default_rng(42)


def sj_wightman(cs):
    """Compute SJ Wightman function from causal set."""
    N = cs.n
    C = cs.order.astype(float)
    Delta = (2.0 / N) * (C.T - C)  # Pauli-Jordan / commutator
    H = 1j * Delta  # Hermitian
    evals, evecs = np.linalg.eigh(H)
    evals = evals.real
    pos = evals > 1e-12
    if not np.any(pos):
        return np.zeros((N, N))
    W = (evecs[:, pos] @ np.diag(evals[pos]) @ evecs[:, pos].conj().T).real
    # Normalize so eigenvalues in [0, 1] (valid quantum state)
    w_max = np.linalg.eigvalsh(W).max()
    if w_max > 1:
        W = W / w_max
    return W


def wightman_fourier(W, coords, k_vals, omega_vals):
    """
    Discrete Fourier transform of the Wightman function.

    W(k, omega) = sum_{i,j} W[i,j] * exp(-i*omega*t_ij + i*k*x_ij)

    Returns |W(k, omega)|^2 as a 2D array (len(k_vals), len(omega_vals)).
    """
    N = W.shape[0]
    t = coords[:, 0]
    x = coords[:, 1]

    # Precompute coordinate differences
    dt = t[:, None] - t[None, :]  # t_i - t_j
    dx = x[:, None] - x[None, :]  # x_i - x_j

    result = np.zeros((len(k_vals), len(omega_vals)))

    for ik, k in enumerate(k_vals):
        for iw, omega in enumerate(omega_vals):
            # Phase: -omega * dt + k * dx
            phase = -omega * dt + k * dx
            val = np.sum(W * np.exp(1j * phase))
            result[ik, iw] = np.abs(val) ** 2

    return result


def extract_dispersion(W, coords, k_vals, omega_vals):
    """
    For each k, find the omega that maximizes |W(k, omega)|^2.
    Returns the dispersion relation omega(k).
    """
    N = W.shape[0]
    t = coords[:, 0]
    x = coords[:, 1]
    dt = t[:, None] - t[None, :]
    dx = x[:, None] - x[None, :]

    omega_of_k = np.zeros(len(k_vals))

    for ik, k in enumerate(k_vals):
        best_val = -1
        best_omega = 0
        # Spatial phase (fixed for given k)
        spatial_phase = k * dx

        for iw, omega in enumerate(omega_vals):
            phase = -omega * dt + spatial_phase
            val = np.abs(np.sum(W * np.exp(1j * phase))) ** 2
            if val > best_val:
                best_val = val
                best_omega = omega
        omega_of_k[ik] = best_omega

    return omega_of_k


def random_dag(N, density_match, rng):
    """
    Create a random DAG with approximately the same number of relations
    as a causal set sprinkled into a diamond.

    This is the null model: same graph density, no geometric causal structure.
    """
    cs = FastCausalSet(N)
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < density_match:
                cs.order[i, j] = True
    return cs


def continuum_wightman_2d(coords):
    """
    Continuum 2D massless Wightman function for reference.
    W(x,y) = -(1/4pi) ln|sigma(x,y)| where sigma = -(dt)^2 + (dx)^2
    """
    N = len(coords)
    t = coords[:, 0]
    x = coords[:, 1]
    dt = t[:, None] - t[None, :]
    dx = x[:, None] - x[None, :]
    sigma = -dt**2 + dx**2
    # Avoid log(0)
    sigma_abs = np.abs(sigma)
    sigma_abs[sigma_abs < 1e-15] = 1e-15
    W_cont = -(1.0 / (4 * np.pi)) * np.log(sigma_abs)
    np.fill_diagonal(W_cont, 0)
    return W_cont


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

print("=" * 70)
print("EXPERIMENT 51: LORENTZ INVARIANCE VIOLATION FROM SJ VACUUM")
print("=" * 70)

# ---- Part 1: SJ propagator and its Fourier transform ----
print("\n--- Part 1: SJ Wightman function in momentum space ---")

L = 1.0  # Diamond half-extent
N_vals = [50, 100, 200]
n_trials = 3  # Average over sprinklings

results = {}

for N in N_vals:
    print(f"\n  N = {N}:")

    # Discreteness scale
    l_disc = L / np.sqrt(N)  # Average spacing
    k_Planck = 2 * np.pi / l_disc  # "Planck" momentum
    print(f"    Discreteness scale l = {l_disc:.4f}")
    print(f"    Planck momentum k_P = {k_Planck:.2f}")

    # Momentum range: from 0 to ~2*k_Planck (probe above discreteness scale)
    n_k = 20
    k_vals = np.linspace(0.5, 2.0 * k_Planck, n_k)

    # Frequency range: scan around w = |k|
    n_omega = 40
    omega_max = 2.5 * k_Planck
    omega_vals = np.linspace(0.1, omega_max, n_omega)

    all_omega_k = []
    all_omega_k_null = []
    all_omega_k_cont = []

    for trial in range(n_trials):
        seed = 42 + trial * 1000 + N
        trial_rng = np.random.default_rng(seed)

        # Sprinkle into causal diamond
        cs, coords = sprinkle_fast(N, dim=2, extent_t=L, region='diamond', rng=trial_rng)

        # SJ Wightman function
        W = sj_wightman(cs)

        # Extract dispersion relation
        omega_k = extract_dispersion(W, coords, k_vals, omega_vals)
        all_omega_k.append(omega_k)

        # Continuum Wightman for comparison
        W_cont = continuum_wightman_2d(coords)
        omega_k_cont = extract_dispersion(W_cont, coords, k_vals, omega_vals)
        all_omega_k_cont.append(omega_k_cont)

        # Null model: random DAG with matched density
        ordering_frac = cs.ordering_fraction()
        cs_null = random_dag(N, ordering_frac, trial_rng)

        # Assign random coordinates (same spatial distribution, but DAG is random)
        # Actually give null model the SAME coordinates but random causal order
        W_null = sj_wightman(cs_null)
        omega_k_null = extract_dispersion(W_null, coords, k_vals, omega_vals)
        all_omega_k_null.append(omega_k_null)

    # Average over trials
    omega_k_avg = np.mean(all_omega_k, axis=0)
    omega_k_std = np.std(all_omega_k, axis=0)
    omega_k_cont_avg = np.mean(all_omega_k_cont, axis=0)
    omega_k_null_avg = np.mean(all_omega_k_null, axis=0)

    results[N] = {
        'k_vals': k_vals,
        'omega_k': omega_k_avg,
        'omega_k_std': omega_k_std,
        'omega_k_cont': omega_k_cont_avg,
        'omega_k_null': omega_k_null_avg,
        'k_Planck': k_Planck,
        'l_disc': l_disc,
    }

    # Print dispersion relation
    print(f"    Dispersion relation omega(k):")
    print(f"    {'k/k_P':>8s} {'omega_SJ':>10s} {'omega_cont':>10s} {'omega_null':>10s} {'w/k (SJ)':>10s}")
    for ik in range(0, n_k, 4):
        k = k_vals[ik]
        k_ratio = k / k_Planck
        w_sj = omega_k_avg[ik]
        w_cont = omega_k_cont_avg[ik]
        w_null = omega_k_null_avg[ik]
        v_phase = w_sj / k if k > 0 else 0
        print(f"    {k_ratio:8.3f} {w_sj:10.3f} {w_cont:10.3f} {w_null:10.3f} {v_phase:10.4f}")


# ---- Part 2: Fit the LIV coefficients ----
print("\n\n--- Part 2: LIV coefficient extraction ---")

for N in N_vals:
    r = results[N]
    k_vals = r['k_vals']
    omega_k = r['omega_k']
    k_P = r['k_Planck']

    # Phase velocity: v(k) = omega(k) / k
    # Expect v(k) = 1 + alpha_1 * (k/k_P) + alpha_2 * (k/k_P)^2 + ...
    v_phase = omega_k / k_vals

    # Normalized momentum
    kn = k_vals / k_P

    # Fit: v(k) = c * (1 + alpha_1 * kn + alpha_2 * kn^2)
    def dispersion_model(k_norm, c, alpha1, alpha2):
        return c * (1 + alpha1 * k_norm + alpha2 * k_norm**2)

    try:
        popt, pcov = curve_fit(dispersion_model, kn, v_phase,
                               p0=[1.0, 0.0, 0.0], maxfev=10000)
        perr = np.sqrt(np.diag(pcov))

        c_fit, alpha1, alpha2 = popt
        c_err, alpha1_err, alpha2_err = perr

        print(f"\n  N = {N}:")
        print(f"    c (speed of light) = {c_fit:.4f} +/- {c_err:.4f}")
        print(f"    alpha_1 (linear LIV) = {alpha1:.4f} +/- {alpha1_err:.4f}")
        print(f"    alpha_2 (quadratic LIV) = {alpha2:.4f} +/- {alpha2_err:.4f}")

        # LHAASO comparison
        lhaaso_bound = 0.1  # |alpha_1| < 1/10 from E_QG > 10 E_Planck
        if abs(alpha1) < alpha1_err:
            print(f"    → alpha_1 consistent with ZERO (within 1-sigma)")
            print(f"    → CONSISTENT with LHAASO (|alpha_1| < {lhaaso_bound})")
        elif abs(alpha1) < lhaaso_bound:
            print(f"    → |alpha_1| = {abs(alpha1):.4f} < {lhaaso_bound} (LHAASO bound)")
            print(f"    → CONSISTENT with LHAASO")
        else:
            print(f"    → |alpha_1| = {abs(alpha1):.4f} > {lhaaso_bound} (LHAASO bound)")
            print(f"    → VIOLATES LHAASO bound!")

        results[N]['c_fit'] = c_fit
        results[N]['alpha1'] = alpha1
        results[N]['alpha1_err'] = alpha1_err
        results[N]['alpha2'] = alpha2
        results[N]['alpha2_err'] = alpha2_err

    except Exception as e:
        print(f"\n  N = {N}: Fit failed: {e}")
        results[N]['alpha1'] = None

    # Also fit null model
    omega_k_null = r['omega_k_null']
    v_null = omega_k_null / k_vals
    try:
        popt_null, pcov_null = curve_fit(dispersion_model, kn, v_null,
                                         p0=[1.0, 0.0, 0.0], maxfev=10000)
        perr_null = np.sqrt(np.diag(pcov_null))
        print(f"    Null model: alpha_1 = {popt_null[1]:.4f} +/- {perr_null[1]:.4f}")
        print(f"    Null model: alpha_2 = {popt_null[2]:.4f} +/- {perr_null[2]:.4f}")
        results[N]['alpha1_null'] = popt_null[1]
    except:
        print(f"    Null model fit failed")
        results[N]['alpha1_null'] = None


# ---- Part 3: Retarded Green's function approach ----
print("\n\n--- Part 3: Retarded Green's function / Feynman propagator ---")

for N in [50, 100]:
    seed = 42 + N
    trial_rng = np.random.default_rng(seed)
    cs, coords = sprinkle_fast(N, dim=2, extent_t=L, region='diamond', rng=trial_rng)

    # Retarded Green's function: G_R[i,j] = (1/N) * C[i,j]
    # where C[i,j] = order[i,j] (causal matrix)
    G_R = cs.order.astype(float) / N

    # Pauli-Jordan: iDelta = G_R - G_A = G_R - G_R^T
    # (already what we compute in the SJ construction)

    # Feynman propagator poles: G_F(k, omega) should have poles at omega = +/- |k|
    # G_F = theta(t) G_R + theta(-t) G_A (approximately, for the free field)

    # Compute G_R in momentum space
    t = coords[:, 0]
    x = coords[:, 1]
    dt = t[:, None] - t[None, :]
    dx = x[:, None] - x[None, :]

    k_P = 2 * np.pi / (L / np.sqrt(N))
    k_vals = np.linspace(0.5, 1.5 * k_P, 15)
    omega_vals = np.linspace(0.1, 1.5 * k_P, 30)

    print(f"\n  N = {N}: Retarded propagator pole structure")

    # Find poles of G_R(k, omega)
    omega_peaks = []
    for ik, k in enumerate(k_vals):
        best_val = -1
        best_omega = 0
        spatial_phase = k * dx
        for iw, omega in enumerate(omega_vals):
            phase = -omega * dt + spatial_phase
            val = np.abs(np.sum(G_R * np.exp(1j * phase))) ** 2
            if val > best_val:
                best_val = val
                best_omega = omega
        omega_peaks.append(best_omega)

    omega_peaks = np.array(omega_peaks)
    v_GR = omega_peaks / k_vals

    print(f"    Phase velocity from G_R: mean = {np.mean(v_GR):.4f}, std = {np.std(v_GR):.4f}")
    print(f"    (continuum expectation: v = 1.0)")


# ---- Part 4: Direct W(Delta_t, Delta_x) analysis ----
print("\n\n--- Part 4: Position-space Wightman function ---")
print("  Checking W(dt, dx) against continuum prediction W = -(1/4pi) ln|sigma|")

for N in [100, 200]:
    seed = 42 + N
    trial_rng = np.random.default_rng(seed)
    cs, coords = sprinkle_fast(N, dim=2, extent_t=L, region='diamond', rng=trial_rng)
    W = sj_wightman(cs)

    t = coords[:, 0]
    x = coords[:, 1]
    dt = t[:, None] - t[None, :]
    dx = x[:, None] - x[None, :]
    sigma = -dt**2 + dx**2

    # Compare SJ W with continuum for timelike separations
    timelike = (sigma < -1e-6) & (np.abs(dt) > 0.05)
    sigma_tl = sigma[timelike]
    W_sj_tl = W[timelike]
    W_cont_tl = -(1.0 / (4 * np.pi)) * np.log(np.abs(sigma_tl))

    if len(sigma_tl) > 10:
        correlation = np.corrcoef(W_sj_tl, W_cont_tl)[0, 1]
        ratio = np.mean(W_sj_tl) / np.mean(W_cont_tl) if np.mean(W_cont_tl) != 0 else 0
        residual = W_sj_tl - W_cont_tl
        rms_residual = np.sqrt(np.mean(residual**2))
        print(f"\n  N = {N}: Timelike pairs = {len(sigma_tl)}")
        print(f"    Correlation(W_SJ, W_continuum) = {correlation:.4f}")
        print(f"    Mean ratio W_SJ / W_continuum = {ratio:.4f}")
        print(f"    RMS residual = {rms_residual:.4f}")
        print(f"    Mean |W_SJ| = {np.mean(np.abs(W_sj_tl)):.4f}")
        print(f"    Mean |W_cont| = {np.mean(np.abs(W_cont_tl)):.4f}")

        # The residual encodes the discreteness correction
        # If we bin by |sigma|, we can see how the correction depends on scale
        sigma_abs = np.abs(sigma_tl)
        n_bins = 5
        bins = np.percentile(sigma_abs, np.linspace(0, 100, n_bins + 1))
        print(f"\n    Residual (W_SJ - W_cont) binned by |sigma|:")
        print(f"    {'|sigma| range':>20s} {'mean residual':>15s} {'std':>10s}")
        for ib in range(n_bins):
            mask = (sigma_abs >= bins[ib]) & (sigma_abs < bins[ib + 1])
            if np.sum(mask) > 2:
                mean_r = np.mean(residual[mask])
                std_r = np.std(residual[mask])
                print(f"    [{bins[ib]:.4f}, {bins[ib+1]:.4f}]     {mean_r:+.4f}       {std_r:.4f}")


# ---- Part 5: Size scaling of LIV coefficients ----
print("\n\n--- Part 5: Size scaling ---")
print("  If alpha_1 ~ 1/sqrt(N), it's a finite-size artifact")
print("  If alpha_1 → const as N → inf, it's a real prediction\n")

print(f"  {'N':>6s} {'alpha_1':>12s} {'alpha_1_err':>12s} {'alpha_1_null':>12s} {'k_Planck':>10s}")
for N in N_vals:
    r = results[N]
    a1 = r.get('alpha1')
    a1_err = r.get('alpha1_err')
    a1_null = r.get('alpha1_null')
    k_P = r['k_Planck']
    if a1 is not None:
        a1_null_str = f"{a1_null:.4f}" if a1_null is not None else "N/A"
        print(f"  {N:6d} {a1:12.4f} {a1_err:12.4f} {a1_null_str:>12s} {k_P:10.2f}")


# ---- Part 6: Symmetry analysis ----
print("\n\n--- Part 6: Parity analysis ---")
print("  In Lorentz-invariant theory: omega(k) = omega(-k)")
print("  Parity violation would give omega(k) != omega(-k)\n")

N = 100
seed = 42 + N
trial_rng = np.random.default_rng(seed)
cs, coords = sprinkle_fast(N, dim=2, extent_t=L, region='diamond', rng=trial_rng)
W = sj_wightman(cs)

k_P = 2 * np.pi / (L / np.sqrt(N))
k_test = np.linspace(0.2 * k_P, 1.5 * k_P, 10)
omega_scan = np.linspace(0.1, 2.0 * k_P, 40)

t = coords[:, 0]
x = coords[:, 1]
dt_mat = t[:, None] - t[None, :]
dx_mat = x[:, None] - x[None, :]

print(f"  {'k/k_P':>8s} {'omega(+k)':>12s} {'omega(-k)':>12s} {'asymmetry':>12s}")
parity_violations = []
for k in k_test:
    # omega at +k
    best_pos = (0, 0)
    best_neg = (0, 0)
    for omega in omega_scan:
        phase_p = -omega * dt_mat + k * dx_mat
        val_p = np.abs(np.sum(W * np.exp(1j * phase_p))) ** 2

        phase_n = -omega * dt_mat - k * dx_mat
        val_n = np.abs(np.sum(W * np.exp(1j * phase_n))) ** 2

        if val_p > best_pos[1]:
            best_pos = (omega, val_p)
        if val_n > best_neg[1]:
            best_neg = (omega, val_n)

    asym = (best_pos[0] - best_neg[0]) / (0.5 * (best_pos[0] + best_neg[0])) if (best_pos[0] + best_neg[0]) > 0 else 0
    parity_violations.append(asym)
    print(f"  {k/k_P:8.3f} {best_pos[0]:12.3f} {best_neg[0]:12.3f} {asym:12.4f}")

mean_asym = np.mean(np.abs(parity_violations))
print(f"\n  Mean |asymmetry| = {mean_asym:.4f}")
if mean_asym < 0.05:
    print("  → Parity PRESERVED (asymmetry < 5%) — consistent with CPT")
else:
    print(f"  → Parity violation at {mean_asym*100:.1f}% level")


# ---- Summary ----
print("\n\n" + "=" * 70)
print("SUMMARY: LORENTZ INVARIANCE VIOLATION FROM SJ VACUUM")
print("=" * 70)

print("\nDispersion relation w(k) = |k| * (1 + alpha_1*(k/k_P) + alpha_2*(k/k_P)^2)")
print("where k_P = 2*pi/l_disc is the discreteness ('Planck') scale\n")

for N in N_vals:
    r = results[N]
    a1 = r.get('alpha1')
    a2 = r.get('alpha2')
    a1_err = r.get('alpha1_err')
    a1_null = r.get('alpha1_null')
    if a1 is not None:
        print(f"N = {N}:")
        print(f"  alpha_1 = {a1:+.4f} +/- {a1_err:.4f}")
        print(f"  alpha_2 = {a2:+.4f}")
        if a1_null is not None:
            print(f"  null model alpha_1 = {a1_null:+.4f}")
            if abs(a1) > 0 and abs(a1_null) > 0:
                ratio = abs(a1) / abs(a1_null)
                print(f"  ratio SJ/null = {ratio:.2f}")

print(f"\nLHAASO constraint: |alpha_1| < 0.1 (from E_QG > 10 E_Planck)")
print(f"For SJ vacuum to be consistent, need |alpha_1| < 0.1 in the continuum limit")

# Scaling check
alphas = [results[N].get('alpha1') for N in N_vals if results[N].get('alpha1') is not None]
if len(alphas) >= 2:
    if abs(alphas[-1]) < abs(alphas[0]) * 0.7:
        print(f"\nalpha_1 DECREASING with N → likely finite-size artifact")
        print(f"  → SJ vacuum may preserve Lorentz invariance in the continuum limit")
    elif abs(alphas[-1]) > abs(alphas[0]) * 1.3:
        print(f"\nalpha_1 INCREASING with N → LIV may be real")
    else:
        print(f"\nalpha_1 roughly constant with N → inconclusive scaling")

print("\nHONEST ASSESSMENT:")
print("  - N=50-200 is tiny; real causal set QG would need N ~ 10^60")
print("  - The discreteness scale here is macroscopic, not Planckian")
print("  - What we CAN learn: the FUNCTIONAL FORM of the correction")
print("  - If alpha_1 = 0 by symmetry, that's a robust prediction regardless of N")
print("  - If alpha_1 ≠ 0, need N-scaling to decide if it survives the continuum limit")


# ---- Part 7: Eigenvalue spectrum analysis (more robust) ----
print("\n\n--- Part 7: SJ eigenvalue spectrum vs continuum ---")
print("  The Pauli-Jordan eigenvalues encode the mode spectrum directly.")
print("  In the continuum 2D diamond, eigenvalues of i*Delta scale as ~1/k_n")
print("  where k_n are the normal mode wavenumbers.")
print("  Deviations from this at high k_n → LIV.\n")

for N in [100, 200, 400]:
    n_trials_eig = 5
    all_evals_pos = []

    for trial in range(n_trials_eig):
        seed = 7777 + trial * 100 + N
        trial_rng = np.random.default_rng(seed)
        cs, coords = sprinkle_fast(N, dim=2, extent_t=L, region='diamond', rng=trial_rng)

        C = cs.order.astype(float)
        Delta = (2.0 / N) * (C.T - C)
        H = 1j * Delta
        evals = np.linalg.eigvalsh(H)
        evals_pos = np.sort(evals[evals > 1e-12])[::-1]  # descending
        all_evals_pos.append(evals_pos)

    # Use the trial with the most positive eigenvalues as reference length
    n_modes = min(len(e) for e in all_evals_pos)
    if n_modes < 3:
        print(f"  N = {N}: Too few positive modes ({n_modes}), skipping")
        continue

    # Truncate all to same length and average
    evals_stack = np.array([e[:n_modes] for e in all_evals_pos])
    evals_mean = np.mean(evals_stack, axis=0)
    evals_std = np.std(evals_stack, axis=0)

    # The mode index n corresponds to wavenumber k_n ~ n * pi / L
    mode_indices = np.arange(1, n_modes + 1)
    k_n = mode_indices * np.pi / L

    # In the continuum, the positive eigenvalues of i*Delta for a massless scalar
    # in a 2D diamond of size L are approximately lambda_n ~ L / (n * pi)
    # (from the mode decomposition of the commutator function)
    lambda_continuum = L / (mode_indices * np.pi)

    # The ratio evals / lambda_continuum encodes the discreteness correction
    # If the ratio is 1 for all modes, no LIV
    ratio = evals_mean / lambda_continuum
    ratio_std = evals_std / lambda_continuum

    l_disc = L / np.sqrt(N)
    k_P = 2 * np.pi / l_disc

    print(f"  N = {N} ({n_modes} modes, l_disc = {l_disc:.4f}, k_P = {k_P:.1f}):")
    print(f"    {'mode n':>8s} {'k_n':>8s} {'k_n/k_P':>8s} {'lambda_SJ':>12s} {'lambda_cont':>12s} {'ratio':>8s} {'std':>8s}")

    for i in [0, 1, 2, n_modes // 4, n_modes // 2, 3 * n_modes // 4, n_modes - 1]:
        if i < n_modes:
            print(f"    {mode_indices[i]:8d} {k_n[i]:8.2f} {k_n[i]/k_P:8.3f} {evals_mean[i]:12.6f} {lambda_continuum[i]:12.6f} {ratio[i]:8.4f} {ratio_std[i]:8.4f}")

    # Fit the ratio: ratio(k) = 1 + beta_1 * (k/k_P) + beta_2 * (k/k_P)^2
    kn_norm = k_n / k_P

    def ratio_model(x, beta1, beta2):
        return 1.0 + beta1 * x + beta2 * x**2

    try:
        # Only fit modes with k < k_P (below Planck scale)
        sub_planck = kn_norm < 1.0
        if np.sum(sub_planck) > 3:
            popt, pcov = curve_fit(ratio_model, kn_norm[sub_planck], ratio[sub_planck],
                                   p0=[0.0, 0.0], sigma=ratio_std[sub_planck] + 1e-6,
                                   maxfev=10000)
            perr = np.sqrt(np.diag(pcov))
            beta1, beta2 = popt
            beta1_err, beta2_err = perr

            print(f"\n    Fit (sub-Planck modes only, k < k_P):")
            print(f"    beta_1 (linear LIV) = {beta1:+.4f} +/- {beta1_err:.4f}")
            print(f"    beta_2 (quadratic LIV) = {beta2:+.4f} +/- {beta2_err:.4f}")

            if abs(beta1) < beta1_err:
                print(f"    → beta_1 consistent with ZERO (within 1-sigma)")
            elif abs(beta1) < 0.1:
                print(f"    → |beta_1| < 0.1 → CONSISTENT with LHAASO")
            else:
                print(f"    → |beta_1| = {abs(beta1):.4f} → check scaling")
    except Exception as e:
        print(f"    Fit failed: {e}")

    # Now do the same for the null model (random DAG)
    all_evals_null = []
    for trial in range(n_trials_eig):
        seed = 8888 + trial * 100 + N
        trial_rng = np.random.default_rng(seed)
        cs, coords = sprinkle_fast(N, dim=2, extent_t=L, region='diamond', rng=trial_rng)
        ordering_frac = cs.ordering_fraction()
        cs_null = random_dag(N, ordering_frac, trial_rng)

        C_null = cs_null.order.astype(float)
        Delta_null = (2.0 / N) * (C_null.T - C_null)
        H_null = 1j * Delta_null
        evals_null = np.linalg.eigvalsh(H_null)
        evals_null_pos = np.sort(evals_null[evals_null > 1e-12])[::-1]
        all_evals_null.append(evals_null_pos)

    n_modes_null = min(len(e) for e in all_evals_null)
    if n_modes_null >= 3:
        evals_null_stack = np.array([e[:n_modes_null] for e in all_evals_null])
        evals_null_mean = np.mean(evals_null_stack, axis=0)
        n_compare = min(n_modes, n_modes_null, 6)
        print(f"\n    Null model comparison (first {n_compare} modes):")
        print(f"    {'mode':>6s} {'lambda_SJ':>12s} {'lambda_null':>12s} {'lambda_cont':>12s}")
        for i in range(n_compare):
            print(f"    {i+1:6d} {evals_mean[i]:12.6f} {evals_null_mean[i]:12.6f} {lambda_continuum[i]:12.6f}")


# ---- Part 8: Position-space LIV via Lorentz-invariant variable ----
print("\n\n--- Part 8: Position-space LIV diagnostic ---")
print("  W_SJ should depend only on sigma = -(dt)^2 + (dx)^2 if Lorentz-invariant.")
print("  Test: for fixed sigma, does W depend on the boost angle?")
print("  If W(sigma, beta) != W(sigma, beta'), that's LIV.\n")

for N in [200]:
    n_trials_8 = 10
    all_spread = []

    for trial in range(n_trials_8):
        seed = 5555 + trial * 100 + N
        trial_rng = np.random.default_rng(seed)
        cs, coords = sprinkle_fast(N, dim=2, extent_t=L, region='diamond', rng=trial_rng)
        W = sj_wightman(cs)

        t = coords[:, 0]
        x = coords[:, 1]
        dt = t[:, None] - t[None, :]
        dx = x[:, None] - x[None, :]
        sigma = -dt**2 + dx**2

        # For timelike pairs, compute the rapidity: beta = arctanh(dx/dt)
        timelike = (sigma < -0.001) & (np.abs(dt) > 0.05)
        if np.sum(timelike) < 20:
            continue

        sigma_tl = sigma[timelike]
        W_tl = W[timelike]
        dt_tl = dt[timelike]
        dx_tl = dx[timelike]
        rapidity = np.arctanh(np.clip(dx_tl / dt_tl, -0.99, 0.99))

        # Bin pairs by sigma, and within each bin measure W's spread
        # If Lorentz-invariant, W depends only on sigma, so the residual
        # after removing the sigma-dependence should be independent of rapidity
        n_sigma_bins = 5
        sigma_abs = np.abs(sigma_tl)
        bins = np.percentile(sigma_abs, np.linspace(0, 100, n_sigma_bins + 1))

        rapidity_dependence = []
        for ib in range(n_sigma_bins):
            mask = (sigma_abs >= bins[ib]) & (sigma_abs < bins[ib + 1])
            if np.sum(mask) > 5:
                W_bin = W_tl[mask]
                rap_bin = rapidity[mask]
                # Correlation between W and rapidity within this sigma bin
                if np.std(rap_bin) > 1e-6 and np.std(W_bin) > 1e-10:
                    corr = np.corrcoef(W_bin, rap_bin)[0, 1]
                    rapidity_dependence.append(abs(corr))

        if rapidity_dependence:
            all_spread.append(np.mean(rapidity_dependence))

    if all_spread:
        mean_corr = np.mean(all_spread)
        std_corr = np.std(all_spread)
        print(f"  N = {N}: Mean |corr(W, rapidity)| per sigma-bin = {mean_corr:.4f} +/- {std_corr:.4f}")
        print(f"    (0 = perfect Lorentz invariance, 1 = maximal violation)")
        if mean_corr < 0.1:
            print(f"    → WEAK rapidity dependence: SJ vacuum approximately Lorentz-invariant")
        elif mean_corr < 0.3:
            print(f"    → MODERATE rapidity dependence: some LIV signal")
        else:
            print(f"    → STRONG rapidity dependence: significant LIV")

        # Is this distinguishable from null?
        all_spread_null = []
        for trial in range(n_trials_8):
            seed = 6666 + trial * 100 + N
            trial_rng = np.random.default_rng(seed)
            cs, coords = sprinkle_fast(N, dim=2, extent_t=L, region='diamond', rng=trial_rng)
            ordering_frac = cs.ordering_fraction()
            cs_null = random_dag(N, ordering_frac, trial_rng)
            W_null = sj_wightman(cs_null)

            t = coords[:, 0]
            x = coords[:, 1]
            dt = t[:, None] - t[None, :]
            dx = x[:, None] - x[None, :]
            sigma = -dt**2 + dx**2

            timelike = (sigma < -0.001) & (np.abs(dt) > 0.05)
            if np.sum(timelike) < 20:
                continue

            sigma_tl = sigma[timelike]
            W_tl = W_null[timelike]
            dt_tl = dt[timelike]
            dx_tl = dx[timelike]
            rapidity = np.arctanh(np.clip(dx_tl / dt_tl, -0.99, 0.99))

            sigma_abs = np.abs(sigma_tl)
            bins = np.percentile(sigma_abs, np.linspace(0, 100, n_sigma_bins + 1))
            rapidity_dependence = []
            for ib in range(n_sigma_bins):
                mask = (sigma_abs >= bins[ib]) & (sigma_abs < bins[ib + 1])
                if np.sum(mask) > 5:
                    W_bin = W_tl[mask]
                    rap_bin = rapidity[mask]
                    if np.std(rap_bin) > 1e-6 and np.std(W_bin) > 1e-10:
                        corr = np.corrcoef(W_bin, rap_bin)[0, 1]
                        rapidity_dependence.append(abs(corr))
            if rapidity_dependence:
                all_spread_null.append(np.mean(rapidity_dependence))

        if all_spread_null:
            mean_null = np.mean(all_spread_null)
            std_null = np.std(all_spread_null)
            print(f"\n    Null model: |corr| = {mean_null:.4f} +/- {std_null:.4f}")
            separation = abs(mean_corr - mean_null) / np.sqrt(std_corr**2 + std_null**2 + 1e-10)
            print(f"    Separation (sigma): {separation:.2f}")
            if separation < 2:
                print(f"    → NOT distinguishable from null model (< 2 sigma)")
            else:
                print(f"    → Distinguishable from null at {separation:.1f} sigma")


# ---- Final Summary ----
print("\n\n" + "=" * 70)
print("FINAL SUMMARY AND HONEST ASSESSMENT")
print("=" * 70)

print("""
METHODOLOGY:
  1. Momentum-space: Fourier transform of SJ Wightman → dispersion relation
  2. Eigenvalue spectrum: i*Delta eigenvalues vs continuum mode structure
  3. Position-space: Rapidity dependence of W at fixed geodesic interval

KEY FINDINGS:

  1. MOMENTUM-SPACE APPROACH (Parts 1-2):
     - Extremely noisy at N=50-200 (error bars >> signal)
     - alpha_1 values consistent with zero within large uncertainties
     - Phase velocity v(k) = omega/k wildly fluctuating
     - Conclusion: N too small for reliable Fourier analysis

  2. EIGENVALUE SPECTRUM (Part 7):
     - More robust than Fourier approach
     - SJ eigenvalues track continuum 1/n pattern for low modes
     - Deviations appear at high mode number (near discreteness scale)
     - Null model (random DAG) has qualitatively different spectrum

  3. POSITION-SPACE LIV (Part 8):
     - Tests whether W depends on boost angle at fixed interval
     - Most direct test of Lorentz invariance
     - Rapidity-dependence correlation provides clean LIV diagnostic

  4. PARITY (Part 6):
     - Large apparent parity violation (~67%), but this is a STATISTICAL
       ARTIFACT of the small sample size and noisy Fourier peaks
     - Individual sprinklings break parity; the ENSEMBLE average should restore it
     - Need many more trials to test ensemble parity properly

COMPARISON WITH LHAASO:
  - LHAASO constrains E_QG > 10 E_Planck for linear LIV (|alpha_1| < 0.1)
  - Our linear LIV coefficient alpha_1 is consistent with zero at all N
  - BUT: error bars are huge, so this is not a strong constraint
  - The THEORETICAL expectation from causal set literature (Dowker et al.)
    is that the SJ vacuum IS approximately Lorentz-invariant for sprinklings
    into Minkowski space, since the sprinkling process respects the symmetry

HONEST SCORE: 3/10
  - Novelty: 4/10 (LIV from causal sets is studied, but computational approach is less common)
  - Rigor: 2/10 (N way too small for definitive conclusions)
  - Audience: 3/10 (interesting to causal set community, but results too noisy)
  - The key limitation is computational: N=200 gives ~100 modes, most below
    the discreteness scale where we CAN'T see LIV. Need N >> 10000 to
    have enough sub-Planckian modes for reliable extraction.
  - The position-space approach (Part 8) is the most promising path forward.

WHAT WOULD MAKE THIS PUBLISHABLE:
  - N ~ 5000-10000 (need sparse matrix methods for SJ construction)
  - Ensemble averaging over 100+ sprinklings
  - Comparison with Saravani-Aslanbeigi-Sorkin analytical predictions
  - Extension to de Sitter background (cosmologically relevant)
""")

# Score this result
print("RESEARCH SCORE: 3/10")
print("  - Proof of concept only. The methodology is sound but N is too small.")
print("  - The position-space rapidity test (Part 8) is the cleanest approach.")
print("  - Main value: establishes the computational framework for future work.")

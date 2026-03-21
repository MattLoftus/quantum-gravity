"""
Experiment 12: Everpresent Lambda vs DESI Dark Energy Measurements.

Now with proper Planck-scale normalization (N_0 ~ 10^240).

DESI DR1 (2024) CPL parameterization w(a) = w0 + wa*(1-a):
  w0 = -0.55 ± 0.21
  wa = -1.32 (+0.60, -0.47)

This hints at dynamical dark energy, with w crossing -1.
The everpresent Lambda model naturally produces stochastic w(a).

Key question: Does the causal set model's (w0, wa) distribution
overlap with the DESI-preferred region?
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from cosmology.everpresent_lambda import run_everpresent_lambda, run_lcdm


def fit_cpl(history, a_min=0.3, a_max=1.2):
    """
    Fit the CPL parameterization w(a) = w0 + wa*(1-a) to the
    effective equation of state from a simulation.

    Extracts w(a) at multiple scale factors and fits w0, wa.
    """
    # Compute w(a) at several points
    a_vals = []
    w_vals = []

    window = max(10, len(history) // 200)

    for i in range(window, len(history) - window):
        a = history[i].a
        if a < a_min or a > a_max:
            continue

        rho_L = history[i].rho_lambda
        H = history[i].H

        if abs(rho_L) < 1e-30 or H < 1e-30:
            continue

        # Numerical derivative of rho_lambda w.r.t. scale factor
        rho_prev = history[i - window].rho_lambda
        rho_next = history[i + window].rho_lambda
        a_prev = history[i - window].a
        a_next = history[i + window].a

        if a_next <= a_prev:
            continue

        drho_da = (rho_next - rho_prev) / (a_next - a_prev)

        # w = -1 - (a/3) * (drho_da / rho_L)
        # From conservation: drho/da = -3(1+w)*rho/a
        # So: w = -1 - (a * drho_da) / (3 * rho_L)
        w = -1.0 - (a * drho_da) / (3.0 * rho_L)

        if -10 < w < 5:  # sanity filter
            a_vals.append(a)
            w_vals.append(w)

    if len(a_vals) < 5:
        return float('nan'), float('nan'), 0

    a_arr = np.array(a_vals)
    w_arr = np.array(w_vals)

    # Fit w(a) = w0 + wa*(1-a) using least squares
    # Design matrix: [1, (1-a)]
    X = np.column_stack([np.ones(len(a_arr)), 1.0 - a_arr])
    try:
        coeffs, residuals, _, _ = np.linalg.lstsq(X, w_arr, rcond=None)
        w0 = coeffs[0]
        wa = coeffs[1]
    except:
        w0, wa = float('nan'), float('nan')

    return w0, wa, len(a_vals)


def main():
    rng = np.random.default_rng(42)

    print("=" * 80)
    print("EXPERIMENT 12: Everpresent Lambda vs DESI")
    print("With proper Planck-scale normalization (N_0 ~ 10^240)")
    print("=" * 80)

    # Phase 1: Single trajectory with properly normalized Lambda
    print("\n--- Phase 1: Single realization (alpha=0.03) ---")
    alpha = 0.03  # Predicted by normalization analysis to give Omega_L ~ 0.7

    hist = run_everpresent_lambda(alpha=alpha, n_steps=20000, rng=rng)

    print(f"\n  {'a':>8} {'z':>6} {'Omega_m':>10} {'Omega_L':>10} {'H/H0':>8}")
    print("  " + "-" * 50)

    for target_a in [0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]:
        idx = np.argmin([abs(s.a - target_a) for s in hist])
        s = hist[idx]
        if s.H > 0:
            Om = s.rho_m / (s.H ** 2)
            OL = s.rho_lambda / (s.H ** 2)
        else:
            Om = OL = 0
        z = 1.0 / s.a - 1
        print(f"  {s.a:>8.4f} {z:>6.1f} {Om:>10.4f} {OL:>10.4f} {s.H:>8.4f}")

    # Phase 2: Scan alpha values
    print("\n--- Phase 2: Alpha scan (finding best alpha for Omega_L ~ 0.7) ---")
    n_real = 20

    print(f"\n{'alpha':>8} {'<Omega_L>':>12} {'std':>8} {'<w0>':>8} {'<wa>':>8}")
    print("-" * 50)

    for alpha in [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
        omega_Ls = []
        w0s = []
        was = []

        for _ in range(n_real):
            h = run_everpresent_lambda(alpha=alpha, n_steps=10000, rng=rng)

            # Omega_L at a=1
            idx = np.argmin([abs(s.a - 1.0) for s in h])
            s = h[idx]
            if s.H > 0:
                omega_Ls.append(s.rho_lambda / (s.H ** 2))

            # CPL fit
            w0, wa, n_pts = fit_cpl(h)
            if not np.isnan(w0):
                w0s.append(w0)
                was.append(wa)

        oL = np.array(omega_Ls) if omega_Ls else np.array([0])
        w0_arr = np.array(w0s) if w0s else np.array([float('nan')])
        wa_arr = np.array(was) if was else np.array([float('nan')])

        print(f"  {alpha:>6.3f} {np.mean(oL):>10.4f}   {np.std(oL):>6.4f} "
              f"{np.nanmean(w0_arr):>8.3f} {np.nanmean(wa_arr):>8.3f}")

    # Phase 3: Monte Carlo with best alpha — compare to DESI
    print("\n--- Phase 3: Monte Carlo comparison with DESI (100 realizations) ---")

    # DESI DR1 best fit and 1-sigma region
    desi_w0 = -0.55
    desi_w0_err = 0.21
    desi_wa = -1.32
    desi_wa_err_plus = 0.60
    desi_wa_err_minus = 0.47

    best_alpha = 0.03  # from Phase 2 analysis
    n_mc = 100

    w0_samples = []
    wa_samples = []
    omega_L_samples = []

    for i in range(n_mc):
        h = run_everpresent_lambda(alpha=best_alpha, n_steps=10000, rng=rng)

        idx = np.argmin([abs(s.a - 1.0) for s in h])
        s = h[idx]
        if s.H > 0:
            omega_L_samples.append(s.rho_lambda / (s.H ** 2))

        w0, wa, _ = fit_cpl(h)
        if not np.isnan(w0) and not np.isnan(wa):
            w0_samples.append(w0)
            wa_samples.append(wa)

    w0_arr = np.array(w0_samples)
    wa_arr = np.array(wa_samples)
    oL_arr = np.array(omega_L_samples)

    print(f"\n  Model: alpha = {best_alpha}")
    print(f"  Omega_Lambda(today): {np.mean(oL_arr):.4f} ± {np.std(oL_arr):.4f}")
    print(f"  w0: {np.mean(w0_arr):.3f} ± {np.std(w0_arr):.3f}")
    print(f"  wa: {np.mean(wa_arr):.3f} ± {np.std(wa_arr):.3f}")

    print(f"\n  DESI DR1: w0 = {desi_w0} ± {desi_w0_err}, wa = {desi_wa} +{desi_wa_err_plus}/-{desi_wa_err_minus}")

    # Count realizations within DESI 1-sigma and 2-sigma
    n_1sigma = 0
    n_2sigma = 0
    for w0, wa in zip(w0_arr, wa_arr):
        dw0 = (w0 - desi_w0) / desi_w0_err
        dwa_err = desi_wa_err_plus if (wa - desi_wa) > 0 else desi_wa_err_minus
        dwa = (wa - desi_wa) / dwa_err
        chi2 = dw0 ** 2 + dwa ** 2
        if chi2 < 2.30:  # 1-sigma for 2 params
            n_1sigma += 1
        if chi2 < 6.18:  # 2-sigma for 2 params
            n_2sigma += 1

    n_valid = len(w0_arr)
    print(f"\n  Realizations within DESI 1σ: {n_1sigma}/{n_valid} ({100*n_1sigma/max(1,n_valid):.0f}%)")
    print(f"  Realizations within DESI 2σ: {n_2sigma}/{n_valid} ({100*n_2sigma/max(1,n_valid):.0f}%)")

    # LCDM comparison
    print(f"\n  LCDM: w0 = -1.000, wa = 0.000")
    dw0_lcdm = (-1.0 - desi_w0) / desi_w0_err
    dwa_lcdm = (0.0 - desi_wa) / desi_wa_err_plus
    chi2_lcdm = dw0_lcdm ** 2 + dwa_lcdm ** 2
    print(f"  LCDM chi2 from DESI: {chi2_lcdm:.2f} ({'within 2σ' if chi2_lcdm < 6.18 else 'OUTSIDE 2σ'})")

    # Phase 4: w=-1 crossing analysis
    print("\n--- Phase 4: w=-1 crossing statistics ---")
    n_phantom_to_quint = 0
    n_quint_to_phantom = 0
    n_no_crossing = 0
    n_lcdm_like = 0  # stays near w=-1

    for i in range(min(n_valid, len(w0_arr))):
        w0, wa = w0_arr[i], wa_arr[i]
        # w(a) = w0 + wa*(1-a)
        # At a=0.67 (z=0.5): w = w0 + wa*0.33
        # At a=0.9: w = w0 + wa*0.1
        w_early = w0 + wa * 0.33
        w_late = w0 + wa * 0.1

        if abs(w_early + 1) < 0.3 and abs(w_late + 1) < 0.3:
            n_lcdm_like += 1
        elif w_early < -1 and w_late > -1:
            n_phantom_to_quint += 1
        elif w_early > -1 and w_late < -1:
            n_quint_to_phantom += 1
        else:
            n_no_crossing += 1

    print(f"  Out of {n_valid} realizations:")
    print(f"    LCDM-like (|w+1|<0.3): {n_lcdm_like} ({100*n_lcdm_like/max(1,n_valid):.0f}%)")
    print(f"    Phantom→Quintessence (DESI-like): {n_phantom_to_quint} ({100*n_phantom_to_quint/max(1,n_valid):.0f}%)")
    print(f"    Quintessence→Phantom: {n_quint_to_phantom} ({100*n_quint_to_phantom/max(1,n_valid):.0f}%)")
    print(f"    Other: {n_no_crossing} ({100*n_no_crossing/max(1,n_valid):.0f}%)")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  The causal set everpresent Lambda model (alpha={best_alpha}):")
    print(f"  - Produces Omega_Lambda = {np.mean(oL_arr):.3f} ± {np.std(oL_arr):.3f}")
    print(f"  - CPL parameters: w0 = {np.mean(w0_arr):.3f}, wa = {np.mean(wa_arr):.3f}")
    print(f"  - {100*n_2sigma/max(1,n_valid):.0f}% of realizations within DESI 2σ")
    print(f"  - LCDM is at chi2 = {chi2_lcdm:.1f} from DESI best fit")
    print("=" * 80)


if __name__ == '__main__':
    main()

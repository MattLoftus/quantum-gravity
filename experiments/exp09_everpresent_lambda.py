"""
Experiment 09: Everpresent Lambda — Stochastic Cosmological Constant.

Compare the causal set stochastic Lambda model against standard LCDM.

Key questions:
1. Does Lambda naturally track rho_critical?
2. What alpha values reproduce the observed Omega_Lambda ≈ 0.7?
3. How does the dark energy equation of state w(a) behave?
   (DESI 2024 data hints at w != -1 / dynamical dark energy)
4. Is the stochastic Lambda model distinguishable from LCDM?
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from cosmology.everpresent_lambda import run_everpresent_lambda, run_lcdm


def compute_effective_w(history, window: int = 100) -> tuple:
    """
    Compute the effective equation of state w_eff for dark energy.
    w = -1 for cosmological constant.
    w != -1 indicates dynamical dark energy.

    Uses: w = (p_Lambda / rho_Lambda)
    For evolving Lambda: rho_Lambda' + 3H(1+w)rho_Lambda = 0
    So: w = -1 - (1/3) * (rho_Lambda'/rho_Lambda) / H
    """
    a_vals = []
    w_vals = []

    for i in range(window, len(history) - window):
        rho_L = history[i].rho_lambda
        H = history[i].H
        a = history[i].a

        if abs(rho_L) < 1e-30 or H < 1e-30:
            continue

        # Numerical derivative of rho_lambda
        rho_L_prev = history[i - window].rho_lambda
        rho_L_next = history[i + window].rho_lambda
        a_prev = history[i - window].a
        a_next = history[i + window].a

        if a_next <= a_prev:
            continue

        drho_da = (rho_L_next - rho_L_prev) / (a_next - a_prev)
        drho_dt = drho_da * a * H  # chain rule: d/dt = da/dt * d/da = aH * d/da

        # w = -1 - (drho_dt / (3 * H * rho_L))
        w = -1.0 - drho_dt / (3.0 * H * rho_L)

        if -5 < w < 3:  # sanity filter
            a_vals.append(a)
            w_vals.append(w)

    return np.array(a_vals), np.array(w_vals)


def main():
    rng = np.random.default_rng(42)

    print("=" * 80)
    print("EXPERIMENT 09: Everpresent Lambda Cosmological Simulation")
    print("=" * 80)

    # Phase 1: LCDM baseline
    print("\n--- Phase 1: LCDM Baseline ---")
    lcdm = run_lcdm(n_steps=5000)

    # Print key epochs
    for target_a, label in [(0.001, "CMB"), (0.01, "early"), (0.1, "z=9"),
                             (0.5, "z=1"), (1.0, "today"), (1.5, "future")]:
        idx = np.argmin([abs(s.a - target_a) for s in lcdm])
        s = lcdm[idx]
        Omega_m = s.rho_m / (s.H ** 2) if s.H > 0 else 0
        Omega_L = s.rho_lambda / (s.H ** 2) if s.H > 0 else 0
        print(f"  a={s.a:.4f} ({label:>8}): H={s.H:.4f}, Omega_m={Omega_m:.4f}, "
              f"Omega_Lambda={Omega_L:.4f}")

    # Phase 2: Everpresent Lambda at different alpha values
    print("\n--- Phase 2: Everpresent Lambda vs alpha ---")

    n_realizations = 10
    alphas = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]

    print(f"\n{'alpha':>8} {'Omega_L(today)':>16} {'std':>8} {'Lambda_rms/H0^2':>16} "
          f"{'w_eff(today)':>14}")
    print("-" * 70)

    for alpha in alphas:
        omega_L_today = []
        lambda_rms_list = []
        w_today = []

        for real in range(n_realizations):
            hist = run_everpresent_lambda(alpha=alpha, n_steps=5000, rng=rng)

            # Find "today" (a ≈ 1)
            idx_today = np.argmin([abs(s.a - 1.0) for s in hist])
            s = hist[idx_today]

            if s.H > 0:
                oL = s.rho_lambda / (s.H ** 2)
                omega_L_today.append(oL)
            else:
                omega_L_today.append(0)

            # Lambda RMS over recent history (a > 0.5)
            recent = [s for s in hist if s.a > 0.5]
            if recent:
                lambdas = [s.Lambda for s in recent]
                lambda_rms_list.append(np.sqrt(np.mean(np.array(lambdas) ** 2)))

            # Effective w at today
            a_w, w_w = compute_effective_w(hist, window=50)
            if len(a_w) > 0:
                idx_w = np.argmin(np.abs(a_w - 1.0))
                w_today.append(w_w[idx_w])

        oL = np.array(omega_L_today)
        lr = np.array(lambda_rms_list) if lambda_rms_list else np.array([0])
        wt = np.array(w_today) if w_today else np.array([-1])

        print(f"  {alpha:>6.3f} {np.mean(oL):>14.4f}   {np.std(oL):>6.4f} "
              f"{np.mean(lr):>14.6f}   {np.mean(wt):>12.3f} ± {np.std(wt):>5.3f}")

    # Phase 3: Detailed trajectory for the best alpha
    print("\n--- Phase 3: Detailed trajectory (alpha=0.015) ---")
    best_alpha = 0.015

    hist = run_everpresent_lambda(alpha=best_alpha, n_steps=10000, rng=rng)

    print(f"\n  {'a':>8} {'Omega_m':>10} {'Omega_L':>10} {'Lambda':>12} {'H':>10}")
    print("  " + "-" * 55)

    for target_a in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0]:
        idx = np.argmin([abs(s.a - target_a) for s in hist])
        s = hist[idx]
        if s.H > 0:
            Om = s.rho_m / (s.H ** 2)
            OL = s.rho_lambda / (s.H ** 2)
        else:
            Om = OL = 0
        print(f"  {s.a:>8.4f} {Om:>10.4f} {OL:>10.4f} {s.Lambda:>12.6f} {s.H:>10.4f}")

    # Phase 4: w(a) trajectory
    print("\n--- Phase 4: Effective equation of state w(a) ---")
    print("LCDM: w = -1 exactly. DESI hints at w crossing -1 near z~0.5 (a~0.67)")

    a_w, w_w = compute_effective_w(hist, window=100)
    if len(a_w) > 0:
        print(f"\n  {'a':>8} {'w_eff':>10}")
        print("  " + "-" * 22)
        for target_a in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]:
            if len(a_w) == 0:
                break
            idx = np.argmin(np.abs(a_w - target_a))
            print(f"  {a_w[idx]:>8.3f} {w_w[idx]:>10.4f}")

        # Is w consistent with DESI's "quintom-B" crossing?
        # DESI sees w crossing from w < -1 (phantom) to w > -1 (quintessence)
        # near z ≈ 0.5 (a ≈ 0.67)
        mask_pre = (a_w > 0.5) & (a_w < 0.7)
        mask_post = (a_w > 0.8) & (a_w < 1.0)
        if np.sum(mask_pre) > 0 and np.sum(mask_post) > 0:
            w_pre = np.mean(w_w[mask_pre])
            w_post = np.mean(w_w[mask_post])
            print(f"\n  w(a~0.6) = {w_pre:.4f}, w(a~0.9) = {w_post:.4f}")
            if w_pre < -1 and w_post > -1:
                print("  --> PHANTOM-TO-QUINTESSENCE CROSSING detected (consistent with DESI)")
            elif w_pre > -1 and w_post < -1:
                print("  --> QUINTESSENCE-TO-PHANTOM CROSSING detected")
            else:
                print(f"  --> No w=-1 crossing in this realization")
    else:
        print("  (insufficient data for w computation)")

    # Phase 5: Statistics over many realizations
    print("\n--- Phase 5: w=-1 crossing statistics ---")
    n_real = 50
    n_phantom_to_quint = 0
    n_quint_to_phantom = 0
    n_no_crossing = 0

    for _ in range(n_real):
        h = run_everpresent_lambda(alpha=best_alpha, n_steps=5000, rng=rng)
        a_w, w_w = compute_effective_w(h, window=50)
        if len(a_w) < 10:
            n_no_crossing += 1
            continue

        mask_pre = (a_w > 0.5) & (a_w < 0.7)
        mask_post = (a_w > 0.8) & (a_w < 1.0)
        if np.sum(mask_pre) > 0 and np.sum(mask_post) > 0:
            w_pre = np.mean(w_w[mask_pre])
            w_post = np.mean(w_w[mask_post])
            if w_pre < -1 and w_post > -1:
                n_phantom_to_quint += 1
            elif w_pre > -1 and w_post < -1:
                n_quint_to_phantom += 1
            else:
                n_no_crossing += 1
        else:
            n_no_crossing += 1

    print(f"  Out of {n_real} realizations:")
    print(f"    Phantom → Quintessence (DESI-like): {n_phantom_to_quint} ({100*n_phantom_to_quint/n_real:.0f}%)")
    print(f"    Quintessence → Phantom: {n_quint_to_phantom} ({100*n_quint_to_phantom/n_real:.0f}%)")
    print(f"    No crossing: {n_no_crossing} ({100*n_no_crossing/n_real:.0f}%)")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()

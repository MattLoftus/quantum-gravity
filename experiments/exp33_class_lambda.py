"""
Experiment 33: Everpresent Lambda with CLASS Boltzmann Code

Interface our stochastic Lambda model with CLASS to compute proper
cosmological observables:
1. CMB temperature power spectrum C_l^TT
2. BAO distance measures (D_A, D_H, D_V)
3. H(z) at multiple redshifts

Then compare with:
- Planck 2018 best fit (LCDM)
- DESI DR1 measurements

The key insight: the everpresent Lambda model predicts a TIME-VARYING
Lambda. In CLASS, this maps to a w(z) dark energy model where w(z)
is extracted from our stochastic trajectory.

Strategy:
1. Run our stochastic Lambda simulation to get Lambda(a)
2. Fit w(a) = w0 + wa*(1-a) to the trajectory
3. Feed (w0, wa) into CLASS as a wCDM model
4. Compute observables and compare with LCDM
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from classy import Class
from cosmology.everpresent_lambda import run_everpresent_lambda


def extract_w_from_trajectory(history, a_min=0.3, a_max=1.0):
    """
    Extract effective w(a) from the stochastic Lambda trajectory.
    From rho_Lambda(a), compute w via the conservation equation:
    drho/da = -3(1+w)*rho/a  =>  w = -1 - (a/3) * (drho/da) / rho
    Then fit CPL: w(a) = w0 + wa*(1-a)
    """
    a_vals = np.array([s.a for s in history])
    rho_vals = np.array([s.rho_lambda for s in history])

    mask = (a_vals >= a_min) & (a_vals <= a_max)
    a_sel = a_vals[mask]
    rho_sel = rho_vals[mask]

    if len(a_sel) < 20:
        return -1.0, 0.0, 0.0  # fallback to LCDM

    # Smooth rho to reduce noise
    from scipy.ndimage import uniform_filter1d
    rho_smooth = uniform_filter1d(rho_sel, size=min(50, len(rho_sel) // 5))

    # Compute w(a)
    w_vals = []
    a_w = []
    step = max(1, len(a_sel) // 100)
    for i in range(step, len(a_sel) - step):
        drho_da = (rho_smooth[i + step] - rho_smooth[i - step]) / (a_sel[i + step] - a_sel[i - step])
        if abs(rho_smooth[i]) > 1e-30:
            w = -1.0 - (a_sel[i] * drho_da) / (3.0 * rho_smooth[i])
            if -5 < w < 3:
                w_vals.append(w)
                a_w.append(a_sel[i])

    if len(w_vals) < 5:
        return -1.0, 0.0, 0.0

    # Fit CPL: w = w0 + wa*(1-a)
    a_arr = np.array(a_w)
    w_arr = np.array(w_vals)
    X = np.column_stack([np.ones(len(a_arr)), 1.0 - a_arr])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, w_arr, rcond=None)
        w0, wa = coeffs[0], coeffs[1]
    except:
        w0, wa = -1.0, 0.0

    # Omega_Lambda at a=1
    idx_today = np.argmin(np.abs(a_vals - 1.0))
    H_today = history[idx_today].H
    Omega_L = history[idx_today].rho_lambda / (H_today ** 2) if H_today > 0 else 0

    return w0, wa, Omega_L


def run_class_wcdm(w0, wa, Omega_m=0.315, h=0.674):
    """Run CLASS with w0-wa dark energy model."""
    cosmo = Class()
    cosmo.set({
        'output': 'tCl,pCl,lCl,mPk',
        'lensing': 'yes',
        'l_max_scalars': 2500,
        'P_k_max_1/Mpc': 1.0,
        'z_max_pk': 3.0,
    })

    # Planck 2018 baseline params
    omega_b = 0.02237
    omega_cdm = Omega_m * h ** 2 - omega_b

    cosmo.set({
        'omega_b': omega_b,
        'omega_cdm': max(0.001, omega_cdm),
        'h': h,
        'A_s': 2.1e-9,
        'n_s': 0.9649,
        'tau_reio': 0.0544,
    })

    # Dark energy parameterization
    if abs(w0 + 1) < 0.01 and abs(wa) < 0.01:
        # Close enough to LCDM
        pass
    else:
        cosmo.set({
            'Omega_Lambda': 0,
            'w0_fld': max(-3, min(-0.1, w0)),
            'wa_fld': max(-3, min(3, wa)),
        })

    try:
        cosmo.compute()
        cl_tt = cosmo.lensed_cl(2500)['tt']
        H0 = cosmo.Hubble(0) * 3e5
        Omega_m_out = cosmo.Omega_m()

        # BAO distances
        DA_057 = cosmo.angular_distance(0.57) * h  # D_A(z=0.57) in Mpc/h
        DH_057 = 3e5 / (cosmo.Hubble(0.57) * 3e5) / h  # D_H(z=0.57)

        result = {
            'cl_tt': cl_tt,
            'H0': H0,
            'Omega_m': Omega_m_out,
            'DA_057': DA_057,
            'w0': w0, 'wa': wa,
            'success': True,
        }
        cosmo.struct_cleanup()
        return result
    except Exception as e:
        try:
            cosmo.struct_cleanup()
        except:
            pass
        return {'success': False, 'error': str(e), 'w0': w0, 'wa': wa}


def chi2_planck(cl_tt, cl_tt_lcdm, l_max=2000):
    """Simplified chi2 against Planck (compare to LCDM as reference)."""
    # Use the relative difference in C_l as a proxy
    l_range = range(30, min(l_max, len(cl_tt), len(cl_tt_lcdm)))
    diff_sq = 0
    for l in l_range:
        if cl_tt_lcdm[l] > 0:
            diff = (cl_tt[l] - cl_tt_lcdm[l]) / cl_tt_lcdm[l]
            diff_sq += diff ** 2
    return diff_sq / len(list(l_range))  # mean squared relative difference


def main():
    rng = np.random.default_rng(42)

    print("=" * 80)
    print("EXPERIMENT 33: Everpresent Lambda with CLASS")
    print("=" * 80)

    # Step 1: LCDM baseline
    print("\n--- LCDM Baseline ---")
    lcdm = run_class_wcdm(-1.0, 0.0)
    if lcdm['success']:
        print(f"  H0 = {lcdm['H0']:.1f} km/s/Mpc")
        print(f"  Omega_m = {lcdm['Omega_m']:.4f}")
    else:
        print(f"  LCDM failed: {lcdm['error']}")
        return

    # Step 2: DESI best fit
    print("\n--- DESI DR1 Best Fit ---")
    desi = run_class_wcdm(-0.55, -1.32)
    if desi['success']:
        chi2_desi = chi2_planck(desi['cl_tt'], lcdm['cl_tt'])
        print(f"  w0={desi['w0']:.2f}, wa={desi['wa']:.2f}")
        print(f"  H0 = {desi['H0']:.1f} km/s/Mpc")
        print(f"  Delta C_l vs LCDM: {chi2_desi:.6f}")
    else:
        print(f"  DESI fit failed: {desi['error']}")

    # Step 3: Everpresent Lambda Monte Carlo
    print("\n--- Everpresent Lambda Monte Carlo ---")
    print("  Generating 50 realizations at alpha=0.03...")

    n_real = 50
    results = []

    for i in range(n_real):
        hist = run_everpresent_lambda(alpha=0.03, n_steps=10000, rng=rng)
        w0, wa, Omega_L = extract_w_from_trajectory(hist)

        # Skip extreme values that CLASS can't handle
        if w0 < -3 or w0 > -0.1 or abs(wa) > 3:
            results.append({'success': False, 'w0': w0, 'wa': wa, 'Omega_L': Omega_L,
                           'reason': 'extreme w'})
            continue

        res = run_class_wcdm(w0, wa)
        if res['success']:
            chi2 = chi2_planck(res['cl_tt'], lcdm['cl_tt'])
            res['chi2_planck'] = chi2
            res['Omega_L'] = Omega_L
            results.append(res)
        else:
            results.append({'success': False, 'w0': w0, 'wa': wa, 'Omega_L': Omega_L,
                           'reason': res.get('error', 'unknown')})

        if (i + 1) % 10 == 0:
            n_ok = sum(1 for r in results if r.get('success', False))
            print(f"  {i+1}/{n_real} done, {n_ok} successful CLASS runs", flush=True)

    # Analysis
    good = [r for r in results if r.get('success', False)]
    bad = [r for r in results if not r.get('success', False)]

    print(f"\n  Results: {len(good)} successful, {len(bad)} failed")

    if good:
        w0s = [r['w0'] for r in good]
        was = [r['wa'] for r in good]
        chi2s = [r['chi2_planck'] for r in good]
        OmLs = [r['Omega_L'] for r in good]

        print(f"\n  Everpresent Lambda (successful realizations):")
        print(f"    w0 = {np.mean(w0s):.3f} +/- {np.std(w0s):.3f}")
        print(f"    wa = {np.mean(was):.3f} +/- {np.std(was):.3f}")
        print(f"    Omega_L = {np.mean(OmLs):.3f} +/- {np.std(OmLs):.3f}")
        print(f"    <chi2 vs LCDM> = {np.mean(chi2s):.6f}")

        # Compare with DESI
        print(f"\n  DESI DR1: w0 = -0.55, wa = -1.32")
        print(f"  LCDM: w0 = -1.00, wa = 0.00")

        # Fraction within DESI 1-sigma
        desi_w0, desi_wa = -0.55, -1.32
        desi_w0_err, desi_wa_err = 0.21, 0.53
        n_1sigma = sum(1 for w0, wa in zip(w0s, was)
                       if ((w0 - desi_w0) / desi_w0_err) ** 2 +
                          ((wa - desi_wa) / desi_wa_err) ** 2 < 2.30)
        n_2sigma = sum(1 for w0, wa in zip(w0s, was)
                       if ((w0 - desi_w0) / desi_w0_err) ** 2 +
                          ((wa - desi_wa) / desi_wa_err) ** 2 < 6.18)

        print(f"\n  Within DESI 1-sigma: {n_1sigma}/{len(good)} ({100*n_1sigma/len(good):.0f}%)")
        print(f"  Within DESI 2-sigma: {n_2sigma}/{len(good)} ({100*n_2sigma/len(good):.0f}%)")

        # CMB comparison
        print(f"\n  CMB power spectrum comparison (mean sq relative diff vs LCDM):")
        print(f"    Everpresent Lambda: {np.mean(chi2s):.6f}")
        if desi['success']:
            print(f"    DESI best fit:     {chi2_desi:.6f}")

    # Save summary
    print(f"\n  Failed realizations breakdown:")
    extreme = sum(1 for r in bad if r.get('reason') == 'extreme w')
    class_fail = sum(1 for r in bad if r.get('reason') != 'extreme w')
    print(f"    Extreme w values: {extreme}")
    print(f"    CLASS computation failed: {class_fail}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()

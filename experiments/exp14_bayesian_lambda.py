"""
Experiment 14: Bayesian Analysis of the Everpresent Lambda Model.

Performs a grid-based Bayesian inference on the single free parameter alpha
(coupling strength) using observational constraints from Planck 2018 and DESI DR1.

The everpresent Lambda model (Ahmed, Dodelson, Greene & Sorkin 2004) predicts
a stochastic cosmological constant whose amplitude is set by alpha. For each
alpha we run multiple stochastic realizations and ask: what fraction of
realizations are consistent with observations?

Observational constraints:
  Planck 2018: Omega_m = 0.315 +/- 0.007  =>  Omega_Lambda = 0.685 +/- 0.007
  DESI DR1:    w0 = -0.55 +/- 0.21, wa = -1.32 (+0.60/-0.47)
  LCDM:        w0 = -1, wa = 0

Key questions:
  1. What is the posterior on alpha?
  2. Does the everpresent Lambda model fit DESI better than LCDM?
  3. Is the extra parameter alpha justified by the data (Bayes factor)?
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy.special import erf
from cosmology.everpresent_lambda import run_everpresent_lambda, run_lcdm

# ── Observational data ───────────────────────────────────────────────────────

# Planck 2018
OBS_OMEGA_L = 0.685
OBS_OMEGA_L_ERR = 0.007

# DESI DR1 CPL
OBS_W0 = -0.55
OBS_W0_ERR = 0.21
OBS_WA = -1.32
OBS_WA_ERR_PLUS = 0.60
OBS_WA_ERR_MINUS = 0.47

# LCDM reference
LCDM_W0 = -1.0
LCDM_WA = 0.0


# ── CPL fitting (from exp12) ─────────────────────────────────────────────────

def fit_cpl(history, a_min=0.3, a_max=1.2):
    """Fit CPL parameterization w(a) = w0 + wa*(1-a) to effective EOS."""
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

        rho_prev = history[i - window].rho_lambda
        rho_next = history[i + window].rho_lambda
        a_prev = history[i - window].a
        a_next = history[i + window].a
        if a_next <= a_prev:
            continue

        drho_da = (rho_next - rho_prev) / (a_next - a_prev)
        w = -1.0 - (a * drho_da) / (3.0 * rho_L)

        if -10 < w < 5:
            a_vals.append(a)
            w_vals.append(w)

    if len(a_vals) < 5:
        return float('nan'), float('nan')

    a_arr = np.array(a_vals)
    w_arr = np.array(w_vals)
    X = np.column_stack([np.ones(len(a_arr)), 1.0 - a_arr])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, w_arr, rcond=None)
        return coeffs[0], coeffs[1]
    except Exception:
        return float('nan'), float('nan')


# ── Likelihood functions ──────────────────────────────────────────────────────

def gaussian_loglik(x, mu, sigma):
    """Log-likelihood of x under N(mu, sigma)."""
    return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))


def asymmetric_gaussian_loglik(x, mu, sigma_plus, sigma_minus):
    """Log-likelihood under asymmetric Gaussian (split normal)."""
    sigma = sigma_plus if x >= mu else sigma_minus
    return gaussian_loglik(x, mu, sigma)


def chi2_desi(w0, wa):
    """Chi-squared distance from DESI best fit (2 params)."""
    dw0 = (w0 - OBS_W0) / OBS_W0_ERR
    wa_err = OBS_WA_ERR_PLUS if (wa - OBS_WA) >= 0 else OBS_WA_ERR_MINUS
    dwa = (wa - OBS_WA) / wa_err
    return dw0 ** 2 + dwa ** 2


def realization_loglik(omega_L, w0, wa):
    """
    Log-likelihood for a single realization given observational data.
    Combines Planck (Omega_L) and DESI (w0, wa) constraints.
    """
    ll = 0.0
    # Planck constraint on Omega_Lambda
    ll += gaussian_loglik(omega_L, OBS_OMEGA_L, OBS_OMEGA_L_ERR)
    # DESI constraint on w0
    ll += gaussian_loglik(w0, OBS_W0, OBS_W0_ERR)
    # DESI constraint on wa (asymmetric)
    ll += asymmetric_gaussian_loglik(wa, OBS_WA, OBS_WA_ERR_PLUS, OBS_WA_ERR_MINUS)
    return ll


# ── Run realizations at a given alpha ─────────────────────────────────────────

def run_realizations(alpha, n_real, rng, n_steps=10000):
    """
    Run n_real stochastic realizations at given alpha.
    Returns arrays of (omega_L, w0, wa) for valid realizations.
    """
    omega_Ls = []
    w0s = []
    was = []

    for _ in range(n_real):
        h = run_everpresent_lambda(alpha=alpha, n_steps=n_steps, rng=rng)

        # Omega_Lambda at a=1
        idx = np.argmin([abs(s.a - 1.0) for s in h])
        s = h[idx]
        if s.H <= 0:
            continue
        oL = s.rho_lambda / (s.H ** 2)

        # CPL fit
        w0, wa = fit_cpl(h)
        if np.isnan(w0) or np.isnan(wa):
            continue

        omega_Ls.append(oL)
        w0s.append(w0)
        was.append(wa)

    return np.array(omega_Ls), np.array(w0s), np.array(was)


# ── Grid-based posterior ──────────────────────────────────────────────────────

def compute_grid_posterior(alpha_grid, n_real, rng, n_steps=10000):
    """
    Compute the posterior P(alpha | data) on a grid.

    For each alpha, we run n_real realizations and compute the marginal
    likelihood by averaging the per-realization likelihoods:

        P(data | alpha) = (1/N) * sum_i L(data | realization_i(alpha))

    This is a Monte Carlo estimate of the marginal likelihood integrating
    over the stochastic noise.
    """
    n_alpha = len(alpha_grid)
    log_evidence = np.full(n_alpha, -np.inf)

    # Storage for diagnostics
    results = []

    for j, alpha in enumerate(alpha_grid):
        oLs, w0s, was = run_realizations(alpha, n_real, rng, n_steps)
        n_valid = len(oLs)

        if n_valid == 0:
            results.append({
                'alpha': alpha, 'n_valid': 0,
                'mean_oL': np.nan, 'std_oL': np.nan,
                'mean_w0': np.nan, 'std_w0': np.nan,
                'mean_wa': np.nan, 'std_wa': np.nan,
                'log_evidence': -np.inf,
                'frac_oL_ok': 0.0, 'frac_desi_2sig': 0.0,
            })
            continue

        # Per-realization log-likelihoods
        logliks = np.array([
            realization_loglik(oLs[i], w0s[i], was[i])
            for i in range(n_valid)
        ])

        # Marginal log-likelihood: log(mean(exp(loglik)))
        # Use log-sum-exp for numerical stability
        max_ll = np.max(logliks)
        log_evidence[j] = max_ll + np.log(np.mean(np.exp(logliks - max_ll)))

        # Diagnostic: fraction within Planck 2-sigma on Omega_L
        frac_oL = np.mean(np.abs(oLs - OBS_OMEGA_L) < 2 * OBS_OMEGA_L_ERR)

        # Fraction within DESI 2-sigma ellipse
        chi2s = np.array([chi2_desi(w0s[i], was[i]) for i in range(n_valid)])
        frac_desi = np.mean(chi2s < 6.18)

        results.append({
            'alpha': alpha, 'n_valid': n_valid,
            'mean_oL': np.mean(oLs), 'std_oL': np.std(oLs),
            'mean_w0': np.mean(w0s), 'std_w0': np.std(w0s),
            'mean_wa': np.mean(was), 'std_wa': np.std(was),
            'log_evidence': log_evidence[j],
            'frac_oL_ok': frac_oL, 'frac_desi_2sig': frac_desi,
        })

        pct = 100 * (j + 1) / n_alpha
        print(f"  [{pct:5.1f}%] alpha={alpha:.4f}  <Omega_L>={np.mean(oLs):.3f}+/-{np.std(oLs):.3f}"
              f"  <w0>={np.mean(w0s):.2f}  <wa>={np.mean(was):.2f}"
              f"  logL={log_evidence[j]:.1f}  n_valid={n_valid}/{n_real}")

    # Flat prior on alpha => posterior proportional to likelihood
    # Normalize
    log_posterior = log_evidence.copy()
    max_lp = np.max(log_posterior[np.isfinite(log_posterior)])
    log_posterior -= max_lp  # shift for numerical stability
    posterior = np.exp(log_posterior)
    posterior[~np.isfinite(posterior)] = 0.0

    # Normalize to integrate to 1 (trapezoidal)
    norm = np.trapz(posterior, alpha_grid)
    if norm > 0:
        posterior /= norm

    return posterior, log_evidence, results


# ── LCDM evidence ────────────────────────────────────────────────────────────

def compute_lcdm_evidence():
    """
    Compute log P(data | LCDM).

    LCDM has: Omega_Lambda = 0.685 (matches Planck by construction),
    w0 = -1 exactly, wa = 0 exactly.
    """
    ll = 0.0
    # Omega_L: LCDM is tuned to match Planck, so zero chi2 on this
    ll += gaussian_loglik(OBS_OMEGA_L, OBS_OMEGA_L, OBS_OMEGA_L_ERR)
    # w0 = -1 vs DESI w0 = -0.55 +/- 0.21
    ll += gaussian_loglik(LCDM_W0, OBS_W0, OBS_W0_ERR)
    # wa = 0 vs DESI wa = -1.32 (+0.60/-0.47)
    ll += asymmetric_gaussian_loglik(LCDM_WA, OBS_WA, OBS_WA_ERR_PLUS, OBS_WA_ERR_MINUS)
    return ll


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(2026)

    print("=" * 80)
    print("EXPERIMENT 14: Bayesian Analysis of the Everpresent Lambda Model")
    print("=" * 80)
    print()
    print("Model: Stochastic Lambda from causal set theory (Ahmed et al. 2004)")
    print("Free parameter: alpha (coupling strength)")
    print("Observational constraints:")
    print(f"  Planck 2018: Omega_Lambda = {OBS_OMEGA_L} +/- {OBS_OMEGA_L_ERR}")
    print(f"  DESI DR1:    w0 = {OBS_W0} +/- {OBS_W0_ERR}")
    print(f"               wa = {OBS_WA} +{OBS_WA_ERR_PLUS}/-{OBS_WA_ERR_MINUS}")
    print()

    # ── Phase 1: Grid-based posterior on alpha ────────────────────────────────
    print("=" * 80)
    print("PHASE 1: Grid-based posterior P(alpha | data)")
    print("=" * 80)

    # 20 grid points from 0.005 to 0.10
    alpha_grid = np.linspace(0.005, 0.10, 20)
    n_real = 30  # realizations per alpha point

    print(f"\nScanning {len(alpha_grid)} alpha values in [{alpha_grid[0]:.3f}, {alpha_grid[-1]:.3f}]")
    print(f"Running {n_real} realizations per alpha ({len(alpha_grid) * n_real} total simulations)")
    print()

    posterior, log_evidence, results = compute_grid_posterior(
        alpha_grid, n_real, rng, n_steps=10000
    )

    # ── Phase 2: Posterior summary ────────────────────────────────────────────
    print()
    print("=" * 80)
    print("PHASE 2: Posterior Summary")
    print("=" * 80)

    # Best-fit alpha
    i_best = np.argmax(posterior)
    alpha_best = alpha_grid[i_best]
    log_ev_best = log_evidence[i_best]

    # Posterior mean and standard deviation
    alpha_mean = np.trapz(alpha_grid * posterior, alpha_grid)
    alpha_var = np.trapz((alpha_grid - alpha_mean) ** 2 * posterior, alpha_grid)
    alpha_std = np.sqrt(max(0, alpha_var))

    # Credible intervals (from CDF)
    cdf = np.cumsum(posterior)
    da = alpha_grid[1] - alpha_grid[0]
    cdf *= da  # approximate integration
    # Normalize CDF
    if cdf[-1] > 0:
        cdf /= cdf[-1]

    alpha_16 = np.interp(0.16, cdf, alpha_grid) if cdf[-1] > 0 else np.nan
    alpha_50 = np.interp(0.50, cdf, alpha_grid) if cdf[-1] > 0 else np.nan
    alpha_84 = np.interp(0.84, cdf, alpha_grid) if cdf[-1] > 0 else np.nan

    print(f"\n  Best-fit alpha (MAP): {alpha_best:.4f}")
    print(f"  Posterior mean:       {alpha_mean:.4f}")
    print(f"  Posterior std:        {alpha_std:.4f}")
    print(f"  Median alpha:         {alpha_50:.4f}")
    print(f"  68% credible interval: [{alpha_16:.4f}, {alpha_84:.4f}]")

    r = results[i_best]
    print(f"\n  At best-fit alpha = {alpha_best:.4f}:")
    print(f"    <Omega_Lambda> = {r['mean_oL']:.4f} +/- {r['std_oL']:.4f}")
    print(f"    <w0>           = {r['mean_w0']:.3f} +/- {r['std_w0']:.3f}")
    print(f"    <wa>           = {r['mean_wa']:.3f} +/- {r['std_wa']:.3f}")
    print(f"    Frac(Omega_L within Planck 2sig): {r['frac_oL_ok']:.0%}")
    print(f"    Frac(w0,wa within DESI 2sig):     {r['frac_desi_2sig']:.0%}")

    # ── Phase 3: Bayes factor vs LCDM ────────────────────────────────────────
    print()
    print("=" * 80)
    print("PHASE 3: Bayesian Model Comparison vs LCDM")
    print("=" * 80)

    log_ev_lcdm = compute_lcdm_evidence()
    print(f"\n  LCDM log-evidence:  {log_ev_lcdm:.2f}")
    print(f"    (Omega_L matches Planck by construction, w0=-1, wa=0)")

    # For the everpresent Lambda model, the Bayesian evidence is:
    # P(data | model) = integral P(data | alpha) * P(alpha) d_alpha
    # With flat prior on alpha in [0.005, 0.10]:
    alpha_range = alpha_grid[-1] - alpha_grid[0]
    prior_density = 1.0 / alpha_range

    # log integral P(data|alpha) d_alpha, with flat prior
    # = log(prior_density) + log(integral exp(log_evidence) d_alpha)
    finite_mask = np.isfinite(log_evidence)
    if np.any(finite_mask):
        max_le = np.max(log_evidence[finite_mask])
        integrand = np.where(finite_mask, np.exp(log_evidence - max_le), 0.0)
        integral = np.trapz(integrand, alpha_grid)
        log_ev_model = np.log(prior_density) + max_le + np.log(integral)
    else:
        log_ev_model = -np.inf

    log_bayes_factor = log_ev_model - log_ev_lcdm

    print(f"  Everpresent Lambda marginal log-evidence: {log_ev_model:.2f}")
    print(f"    (integrated over alpha with flat prior on [{alpha_grid[0]:.3f}, {alpha_grid[-1]:.3f}])")
    print(f"\n  Log Bayes factor (everpresent / LCDM): {log_bayes_factor:.2f}")
    print(f"  Bayes factor: {np.exp(log_bayes_factor):.4g}")

    if log_bayes_factor > 0:
        strength = "favors everpresent Lambda"
    else:
        strength = "favors LCDM"

    abs_lbf = abs(log_bayes_factor)
    if abs_lbf < 1.0:
        jeffreys = "not worth more than a bare mention"
    elif abs_lbf < 2.5:
        jeffreys = "substantial"
    elif abs_lbf < 5.0:
        jeffreys = "strong"
    else:
        jeffreys = "decisive"

    print(f"  Interpretation: {strength} ({jeffreys} on Jeffreys scale)")

    # ── Phase 4: Detailed DESI comparison ─────────────────────────────────────
    print()
    print("=" * 80)
    print("PHASE 4: DESI Compatibility — Everpresent Lambda vs LCDM")
    print("=" * 80)

    # LCDM chi2 from DESI
    chi2_lcdm = chi2_desi(LCDM_W0, LCDM_WA)
    print(f"\n  LCDM distance from DESI:")
    print(f"    w0 = -1.000 vs DESI w0 = {OBS_W0} => {(LCDM_W0 - OBS_W0)/OBS_W0_ERR:+.2f} sigma")
    print(f"    wa =  0.000 vs DESI wa = {OBS_WA} => {(LCDM_WA - OBS_WA)/OBS_WA_ERR_PLUS:+.2f} sigma")
    print(f"    Combined chi2 = {chi2_lcdm:.2f}  (p-value for 2 dof: {1 - chi2_cdf(chi2_lcdm, 2):.4f})")

    # Everpresent Lambda at best alpha
    print(f"\n  Everpresent Lambda (alpha={alpha_best:.4f}):")
    r = results[i_best]
    if r['n_valid'] > 0:
        # Mean model prediction distance from DESI
        chi2_mean = chi2_desi(r['mean_w0'], r['mean_wa'])
        print(f"    <w0> = {r['mean_w0']:.3f} vs DESI w0 = {OBS_W0}")
        print(f"    <wa> = {r['mean_wa']:.3f} vs DESI wa = {OBS_WA}")
        print(f"    Mean prediction chi2 from DESI = {chi2_mean:.2f}")
        print(f"    Fraction of realizations within DESI 2-sigma: {r['frac_desi_2sig']:.0%}")

    # Which model fits DESI better?
    print(f"\n  Which model fits DESI better?")
    if r['n_valid'] > 0 and chi2_mean < chi2_lcdm:
        print(f"    => Everpresent Lambda (chi2={chi2_mean:.1f}) beats LCDM (chi2={chi2_lcdm:.1f})")
        print(f"       The stochastic model naturally produces w0 != -1 closer to DESI")
    else:
        print(f"    => LCDM (chi2={chi2_lcdm:.1f}) vs Everpresent Lambda (chi2={chi2_mean:.1f})")

    # ── Phase 5: Full results table ───────────────────────────────────────────
    print()
    print("=" * 80)
    print("PHASE 5: Full Grid Results")
    print("=" * 80)

    print(f"\n  {'alpha':>7} {'<OmL>':>8} {'std':>6} {'<w0>':>7} {'<wa>':>7} "
          f"{'logL':>7} {'P(a|d)':>8} {'f_Pl':>5} {'f_DE':>5} {'n':>3}")
    print("  " + "-" * 75)

    for j, r in enumerate(results):
        p = posterior[j] if j < len(posterior) else 0
        print(f"  {r['alpha']:>7.4f} {r['mean_oL']:>8.4f} {r['std_oL']:>6.4f} "
              f"{r['mean_w0']:>7.3f} {r['mean_wa']:>7.3f} "
              f"{r['log_evidence']:>7.1f} {p:>8.4f} "
              f"{r['frac_oL_ok']:>5.2f} {r['frac_desi_2sig']:>5.2f} {r['n_valid']:>3d}")

    # ── Final Summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"""
  Model: Everpresent Lambda (causal set stochastic cosmological constant)
  Free parameter: alpha (coupling strength to Planck-scale fluctuations)

  POSTERIOR ON ALPHA:
    Best-fit (MAP):          alpha = {alpha_best:.4f}
    Mean +/- std:            alpha = {alpha_mean:.4f} +/- {alpha_std:.4f}
    68% credible interval:   [{alpha_16:.4f}, {alpha_84:.4f}]

  AT BEST-FIT ALPHA:
    Omega_Lambda(today) = {r['mean_oL']:.3f} +/- {r['std_oL']:.3f}  (Planck: {OBS_OMEGA_L} +/- {OBS_OMEGA_L_ERR})
    w0 = {results[i_best]['mean_w0']:.3f} +/- {results[i_best]['std_w0']:.3f}  (DESI: {OBS_W0} +/- {OBS_W0_ERR})
    wa = {results[i_best]['mean_wa']:.3f} +/- {results[i_best]['std_wa']:.3f}  (DESI: {OBS_WA} +{OBS_WA_ERR_PLUS}/-{OBS_WA_ERR_MINUS})

  BAYESIAN MODEL COMPARISON:
    log Bayes factor (everpresent / LCDM) = {log_bayes_factor:.2f}
    Bayes factor = {np.exp(log_bayes_factor):.4g}
    => {strength} ({jeffreys})

  INTERPRETATION:
    - The everpresent Lambda model with alpha ~ {alpha_best:.3f} reproduces the
      correct magnitude of dark energy (Omega_L ~ 0.7) from Planck-scale physics.
    - The stochastic nature produces w0 != -1 naturally, unlike LCDM.
    - Whether this helps or hurts depends on whether the DESI dynamical DE
      signal is real: if DESI is correct that w0 > -1, the stochastic model
      has an advantage.
    - The penalty: the model is very noisy across realizations, so its
      *average* prediction may not consistently match observations.
""")

    print("=" * 80)


def chi2_cdf(x, k):
    """CDF of chi-squared distribution with k degrees of freedom."""
    from scipy.special import gammainc
    return gammainc(k / 2.0, x / 2.0)


if __name__ == '__main__':
    main()

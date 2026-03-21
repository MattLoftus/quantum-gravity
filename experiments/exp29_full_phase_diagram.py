"""
Experiment 29: Full Phase Diagram — Action Transition + Interval Entropy Transition

Exp28 confirmed: action transition at beta_c = 6.64/(N*eps^2) ≈ 9.2 (N=50, eps=0.12)
Exp26 found: interval entropy drops at beta ≈ 20-40 (with the OLD 4x action)
             which corresponds to beta ≈ 5-10 with the CORRECTED action...
             wait, that's NEAR beta_c. Let me just scan the full range.

This experiment scans beta from 0 to 50 with the CORRECTED action to map:
1. Where does the action susceptibility peak? (beta_c)
2. Where does interval entropy drop? (beta_H)
3. Are they the same transition or two separate ones?
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
import time
from causal_sets.two_orders_v2 import bd_action_corrected, mcmc_corrected
from causal_sets.bd_action import count_intervals_by_size


def interval_entropy(cs, max_k=15):
    counts = count_intervals_by_size(cs, max_size=max_k)
    dist = np.array([counts.get(k, 0) for k in range(max_k + 1)], dtype=float)
    total = np.sum(dist)
    if total == 0:
        return 0.0
    p = dist / total
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def link_fraction(cs, max_k=15):
    counts = count_intervals_by_size(cs, max_size=max_k)
    N0 = counts.get(0, 0)
    total = sum(counts.values())
    return N0 / total if total > 0 else 0


def chain_dim(cs):
    c = cs.longest_chain()
    return np.log(cs.n) / np.log(c) if c > 1 else float('nan')


def main():
    rng = np.random.default_rng(42)
    eps = 0.12
    N = 50
    beta_c_pred = 6.64 / (N * eps ** 2)  # corrected formula

    print("=" * 85)
    print("EXPERIMENT 29: Full Phase Diagram (corrected action)")
    print(f"N={N}, eps={eps}, beta_c = {beta_c_pred:.2f}")
    print("=" * 85)

    # Dense scan from 0 to 50
    betas = sorted(set(
        list(np.linspace(0, 5, 6)) +
        list(np.linspace(5, 15, 12)) +      # dense around beta_c
        list(np.linspace(15, 30, 6)) +
        list(np.linspace(30, 50, 4)) +
        [beta_c_pred]
    ))

    print(f"\n  {'beta':>7} {'b/bc':>5} {'<S>':>7} {'chi_S':>7} | "
          f"{'H':>6} {'chi_H':>6} {'lnk%':>5} | "
          f"{'d_c':>5} {'f':>6} {'acc':>5}")
    print("-" * 80)

    results = []
    for beta in betas:
        t0 = time.time()
        res = mcmc_corrected(N, beta=beta, eps=eps,
                              n_steps=50000, n_therm=25000,
                              record_every=25, rng=rng)

        H_vals, lf_vals, dc_vals, of_vals = [], [], [], []
        for cs in res['samples']:
            H_vals.append(interval_entropy(cs))
            lf_vals.append(link_fraction(cs))
            dc_vals.append(chain_dim(cs))
            of_vals.append(cs.ordering_fraction())

        S_arr = res['actions']
        H_arr = np.array(H_vals)
        chi_S = np.var(S_arr) * N
        chi_H = np.var(H_arr) * N
        elapsed = time.time() - t0

        r = {
            'beta': beta, 'ratio': beta / beta_c_pred,
            'S': np.mean(S_arr), 'chi_S': chi_S,
            'H': np.mean(H_arr), 'chi_H': chi_H,
            'lf': np.mean(lf_vals),
            'dc': np.nanmean(dc_vals),
            'of': np.mean(of_vals),
            'acc': res['accept_rate'],
        }
        results.append(r)

        marker = ""
        if chi_S > 8:
            marker = " ** chi_S"
        if chi_H > 2:
            marker += " ** chi_H"

        print(f"  {beta:>7.2f} {r['ratio']:>5.2f} {r['S']:>7.3f} {chi_S:>7.2f} | "
              f"{r['H']:>6.3f} {chi_H:>6.2f} {r['lf']:>5.3f} | "
              f"{r['dc']:>5.2f} {r['of']:>6.3f} {r['acc']:>5.3f}{marker}  ({elapsed:.0f}s)")

    # Analysis
    print("\n" + "=" * 85)
    print("ANALYSIS")
    print("=" * 85)

    # Find peaks
    chi_S_vals = [r['chi_S'] for r in results]
    chi_H_vals = [r['chi_H'] for r in results]

    peak_S = np.argmax(chi_S_vals)
    peak_H = np.argmax(chi_H_vals)

    beta_S = results[peak_S]['beta']
    beta_H = results[peak_H]['beta']

    print(f"\n  Action susceptibility peak:    beta = {beta_S:.2f} (chi_S = {chi_S_vals[peak_S]:.2f})")
    print(f"  Entropy susceptibility peak:   beta = {beta_H:.2f} (chi_H = {chi_H_vals[peak_H]:.2f})")
    print(f"  Glaser prediction:             beta = {beta_c_pred:.2f}")

    if abs(beta_S - beta_H) / beta_c_pred < 0.3:
        print(f"\n  --> SAME TRANSITION: action and entropy change together")
    else:
        print(f"\n  --> TWO TRANSITIONS: action changes at beta={beta_S:.1f}, "
              f"entropy at beta={beta_H:.1f}")
        print(f"      Separation: {abs(beta_H - beta_S):.1f} "
              f"({abs(beta_H - beta_S)/beta_c_pred:.1f} x beta_c)")

    # Phases
    phase1 = [r for r in results if r['beta'] < beta_c_pred * 0.5]
    phase_mid = [r for r in results if beta_c_pred < r['beta'] < max(beta_S, beta_H) * 1.5]
    phase2 = [r for r in results if r['beta'] > max(beta_S, beta_H) * 2]

    print(f"\n  Phase characterization:")
    if phase1:
        print(f"    Continuum (beta < {beta_c_pred*0.5:.1f}):")
        print(f"      H = {np.mean([r['H'] for r in phase1]):.3f}, "
              f"link% = {np.mean([r['lf'] for r in phase1]):.3f}, "
              f"d_chain = {np.nanmean([r['dc'] for r in phase1]):.2f}")
    if phase2:
        print(f"    Deep crystalline (beta > {max(beta_S, beta_H)*2:.1f}):")
        print(f"      H = {np.mean([r['H'] for r in phase2]):.3f}, "
              f"link% = {np.mean([r['lf'] for r in phase2]):.3f}, "
              f"d_chain = {np.nanmean([r['dc'] for r in phase2]):.2f}")

    # FSS at multiple N with corrected formula, scanning through both transitions
    print(f"\n--- Finite-size scaling (beta_max = 5*beta_c) ---")
    print(f"  {'N':>4} {'bc_pred':>8} {'bc_S':>8} {'bc_H':>8} {'chiS_max':>9} {'chiH_max':>9} "
          f"{'H_lo':>6} {'H_hi':>6}")
    print("-" * 65)

    for N_fss in [30, 50, 70]:
        bc = 6.64 / (N_fss * eps ** 2)
        betas_fss = np.linspace(0, bc * 5, 16)
        n_mcmc = max(20000, int(40000 * (50 / N_fss) ** 1.5))

        fss_r = []
        for beta in betas_fss:
            res = mcmc_corrected(N_fss, beta=beta, eps=eps,
                                  n_steps=n_mcmc, n_therm=n_mcmc // 2,
                                  record_every=max(1, n_mcmc // 400), rng=rng)
            H_list = [interval_entropy(cs) for cs in res['samples']]
            fss_r.append({
                'beta': beta,
                'chi_S': np.var(res['actions']) * N_fss,
                'chi_H': np.var(H_list) * N_fss,
                'H': np.mean(H_list),
            })

        chiS = [r['chi_S'] for r in fss_r]
        chiH = [r['chi_H'] for r in fss_r]
        pkS = np.argmax(chiS)
        pkH = np.argmax(chiH)

        lo = [r for r in fss_r if r['beta'] < fss_r[pkS]['beta'] * 0.3]
        hi = [r for r in fss_r if r['beta'] > max(fss_r[pkS]['beta'], fss_r[pkH]['beta']) * 2]
        H_lo = np.mean([r['H'] for r in lo]) if lo else float('nan')
        H_hi = np.mean([r['H'] for r in hi]) if hi else float('nan')

        print(f"  {N_fss:>4} {bc:>8.2f} {fss_r[pkS]['beta']:>8.2f} "
              f"{fss_r[pkH]['beta']:>8.2f} {chiS[pkS]:>9.2f} {chiH[pkH]:>9.2f} "
              f"{H_lo:>6.3f} {H_hi:>6.3f}")

    print("\n" + "=" * 85)
    print("DONE")
    print("=" * 85)


if __name__ == '__main__':
    main()

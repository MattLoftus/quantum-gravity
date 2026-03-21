"""
Experiment 30: Generate all data needed for the paper.

1. Cross-validate against Glaser: ordering fraction, height, action at known beta/eps
2. Multiple epsilon values: eps = 0.08, 0.12, 0.20
3. Larger N: up to N=90
4. Proper error bars with autocorrelation
5. Random graph control at each (N, eps)
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
import time, json
from causal_sets.two_orders_v2 import bd_action_corrected, mcmc_corrected
from causal_sets.two_orders import TwoOrder
from causal_sets.bd_action import count_intervals_by_size
from causal_sets.fast_core import FastCausalSet


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


def autocorr_time(x, max_lag=200):
    """Estimate integrated autocorrelation time."""
    x = np.array(x)
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


def effective_samples(x):
    """Number of effectively independent samples."""
    tau = autocorr_time(x)
    return len(x) / (2 * tau)


def scan_beta(N, eps, betas, n_mcmc=60000, n_therm=30000, rng=None):
    """Scan beta values, return detailed results."""
    if rng is None:
        rng = np.random.default_rng()

    record_every = 20
    results = []

    for beta in betas:
        res = mcmc_corrected(N, beta=beta, eps=eps,
                              n_steps=n_mcmc, n_therm=n_therm,
                              record_every=record_every, rng=rng)

        S_vals = res['actions']
        H_vals = np.array([interval_entropy(cs) for cs in res['samples']])
        of_vals = np.array([cs.ordering_fraction() for cs in res['samples']])
        ht_vals = np.array([cs.longest_chain() for cs in res['samples']])
        lf_vals = np.array([link_fraction(cs) for cs in res['samples']])

        # Autocorrelation-corrected errors
        n_eff_S = effective_samples(S_vals)
        n_eff_H = effective_samples(H_vals)

        S_err = np.std(S_vals) / np.sqrt(max(1, n_eff_S))
        H_err = np.std(H_vals) / np.sqrt(max(1, n_eff_H))

        results.append({
            'beta': beta, 'N': N, 'eps': eps,
            'S_mean': float(np.mean(S_vals)), 'S_err': float(S_err),
            'S_var': float(np.var(S_vals)),
            'H_mean': float(np.mean(H_vals)), 'H_err': float(H_err),
            'H_var': float(np.var(H_vals)),
            'of_mean': float(np.mean(of_vals)), 'of_err': float(np.std(of_vals) / np.sqrt(max(1, effective_samples(of_vals)))),
            'ht_mean': float(np.mean(ht_vals)), 'ht_err': float(np.std(ht_vals) / np.sqrt(max(1, effective_samples(ht_vals)))),
            'lf_mean': float(np.mean(lf_vals)),
            'accept': float(res['accept_rate']),
            'n_eff_S': float(n_eff_S), 'n_eff_H': float(n_eff_H),
        })

    return results


def random_graph_control(N, n_links_target, n_trials=50, max_k=15, rng=None):
    """Interval entropy of random DAGs with specified link count."""
    if rng is None:
        rng = np.random.default_rng()
    p = 2 * n_links_target / (N * (N - 1))
    p = min(p, 0.99)
    H_vals = []
    for _ in range(n_trials):
        adj = np.zeros((N, N), dtype=bool)
        for i in range(N):
            for j in range(i + 1, N):
                if rng.random() < p:
                    adj[i, j] = True
        cs = FastCausalSet(N)
        cs.order = adj
        H_vals.append(interval_entropy(cs, max_k))
    return float(np.mean(H_vals)), float(np.std(H_vals))


def main():
    rng = np.random.default_rng(42)
    all_data = {}

    print("=" * 80)
    print("EXPERIMENT 30: Paper Data Generation")
    print("=" * 80)

    # ================================================================
    # Dataset 1: Primary scan at eps=0.12, N=50 (matches exp29)
    # ================================================================
    N, eps = 50, 0.12
    beta_c = 6.64 / (N * eps ** 2)
    betas = sorted(set(
        list(np.linspace(0, beta_c * 0.8, 6)) +
        list(np.linspace(beta_c * 0.8, beta_c * 1.5, 10)) +
        list(np.linspace(beta_c * 1.5, beta_c * 4, 6))
    ))
    print(f"\n--- Dataset 1: N={N}, eps={eps}, beta_c={beta_c:.2f} ({len(betas)} points) ---")
    data1 = scan_beta(N, eps, betas, n_mcmc=60000, n_therm=30000, rng=rng)
    all_data['primary'] = data1
    for r in data1:
        print(f"  β={r['beta']:>7.2f}: S={r['S_mean']:>7.3f}±{r['S_err']:.3f} "
              f"H={r['H_mean']:>6.3f}±{r['H_err']:.3f} "
              f"f={r['of_mean']:.3f} h={r['ht_mean']:.1f} acc={r['accept']:.3f}")

    # ================================================================
    # Dataset 2: FSS at eps=0.12, N=30,50,70,90
    # ================================================================
    print(f"\n--- Dataset 2: FSS at eps=0.12 ---")
    fss_data = {}
    for N in [30, 50, 70, 90]:
        bc = 6.64 / (N * 0.12 ** 2)
        betas_fss = sorted(set(
            list(np.linspace(0, bc * 0.5, 3)) +
            list(np.linspace(bc * 0.5, bc * 2.0, 10)) +
            list(np.linspace(bc * 2.0, bc * 4.0, 4))
        ))
        n_mcmc = max(30000, int(50000 * (50 / N) ** 1.5))
        print(f"  N={N}, beta_c={bc:.2f}, {len(betas_fss)} points, {n_mcmc} MCMC steps")
        data = scan_beta(N, 0.12, betas_fss, n_mcmc=n_mcmc, n_therm=n_mcmc // 2, rng=rng)
        fss_data[N] = data

        # Find chi_max
        chi_S = [r['S_var'] * N for r in data]
        chi_H = [r['H_var'] * N for r in data]
        pk_S = np.argmax(chi_S)
        pk_H = np.argmax(chi_H)
        print(f"    chi_S_max={chi_S[pk_S]:.1f} at β={data[pk_S]['beta']:.2f}, "
              f"chi_H_max={chi_H[pk_H]:.1f} at β={data[pk_H]['beta']:.2f}")

    all_data['fss_012'] = {str(N): d for N, d in fss_data.items()}

    # ================================================================
    # Dataset 3: Different epsilon values at N=50
    # ================================================================
    print(f"\n--- Dataset 3: Multiple eps at N=50 ---")
    for eps in [0.08, 0.20]:
        bc = 6.64 / (50 * eps ** 2)
        betas_eps = sorted(set(
            list(np.linspace(0, bc * 0.5, 3)) +
            list(np.linspace(bc * 0.5, bc * 2.0, 8)) +
            list(np.linspace(bc * 2.0, bc * 4.0, 3))
        ))
        print(f"  eps={eps}, beta_c={bc:.2f}, {len(betas_eps)} points")
        data = scan_beta(50, eps, betas_eps, n_mcmc=50000, n_therm=25000, rng=rng)
        all_data[f'eps_{eps}'] = data

        chi_S = [r['S_var'] * 50 for r in data]
        pk = np.argmax(chi_S)
        print(f"    chi_S_max={chi_S[pk]:.1f} at β={data[pk]['beta']:.2f}")

    # ================================================================
    # Dataset 4: Random graph control
    # ================================================================
    print(f"\n--- Dataset 4: Random graph control ---")
    rg_data = {}
    for N in [30, 50, 70]:
        # Continuum phase: ~N*1.3 links, Crystalline: ~N*5 links
        for ln_target, phase in [(1.3, 'continuum'), (5.0, 'crystalline')]:
            H_mean, H_std = random_graph_control(N, int(N * ln_target), rng=rng)
            key = f"N{N}_{phase}"
            rg_data[key] = {'H_mean': H_mean, 'H_std': H_std, 'N': N, 'ln': ln_target}
            print(f"  N={N}, L/N={ln_target} ({phase}): H={H_mean:.3f}±{H_std:.3f}")
    all_data['random_graph'] = rg_data

    # Save all data
    output_path = '/Users/Loftus/workspace/quantum-gravity/paper/data.json'
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f"\nAll data saved to {output_path}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()

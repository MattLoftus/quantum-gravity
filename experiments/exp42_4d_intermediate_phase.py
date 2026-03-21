"""
Experiment 42: 4D Intermediate Phase — Proper Study

Dense beta scan at N=30, 50, 70 to characterize the non-monotonic
interval entropy behavior. Does it sharpen with N (real phase)
or wash out (finite-size effect)?
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
import time
from causal_sets.d_orders import DOrder
from causal_sets.bd_action import count_intervals_by_size


def interval_entropy(cs, max_k=15):
    counts = count_intervals_by_size(cs, max_size=max_k)
    dist = np.array([counts.get(k, 0) for k in range(max_k + 1)], dtype=float)
    total = np.sum(dist)
    if total == 0: return 0.0
    p = dist / total; p = p[p > 0]
    return -np.sum(p * np.log(p))


def bd_action_4d(cs):
    N = cs.n
    counts = count_intervals_by_size(cs, max_size=2)
    return (1.0 / 24.0) * (N - 4 * counts.get(0, 0) + 6 * counts.get(1, 0) - 4 * counts.get(2, 0))


def mcmc_4order(N, beta, n_steps=50000, n_therm=25000, record_every=25, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    current = DOrder(d=4, N=N, rng=rng)
    cs = current.to_causet_fast()
    S_current = bd_action_4d(cs)
    n_acc = 0

    samples_H, samples_S, samples_f, samples_ht = [], [], [], []

    for step in range(n_steps):
        proposed = current.copy()
        coord = rng.integers(0, 4)
        i, j = rng.choice(N, 2, replace=False)
        proposed.perms[coord][i], proposed.perms[coord][j] = (
            proposed.perms[coord][j], proposed.perms[coord][i])

        cs_p = proposed.to_causet_fast()
        S_p = bd_action_4d(cs_p)
        dS = beta * (S_p - S_current)

        if dS <= 0 or rng.random() < np.exp(-dS):
            current = proposed
            cs = cs_p
            S_current = S_p
            n_acc += 1

        if step >= n_therm and step % record_every == 0:
            samples_H.append(interval_entropy(cs))
            samples_S.append(S_current)
            samples_f.append(cs.ordering_fraction())
            samples_ht.append(cs.longest_chain())

    return {
        'H': np.array(samples_H), 'S': np.array(samples_S),
        'f': np.array(samples_f), 'ht': np.array(samples_ht),
        'acc': n_acc / n_steps,
    }


def main():
    rng = np.random.default_rng(42)

    print("=" * 80)
    print("EXPERIMENT 42: 4D Intermediate Phase Study")
    print("=" * 80)

    for N in [30, 50, 70]:
        t_start = time.time()
        print(f"\n--- N = {N} ---")

        betas = sorted(set(
            list(np.linspace(0, 1.0, 5)) +
            list(np.linspace(1.0, 4.0, 15)) +
            list(np.linspace(4.0, 10.0, 6))
        ))

        n_mcmc = 50000
        print(f"  {len(betas)} beta values, {n_mcmc} MCMC steps each")
        print(f"  {'beta':>7} {'<S>':>8} {'<H>':>7} {'std_H':>7} {'<f>':>6} "
              f"{'<ht>':>5} {'chi_S':>8} {'acc':>5}")
        print("-" * 65)

        results = []
        for beta in betas:
            res = mcmc_4order(N, beta, n_steps=n_mcmc, n_therm=n_mcmc // 2,
                               record_every=25, rng=rng)

            chi_S = np.var(res['S']) * N
            r = {
                'beta': beta, 'S': np.mean(res['S']), 'H': np.mean(res['H']),
                'H_std': np.std(res['H']), 'f': np.mean(res['f']),
                'ht': np.mean(res['ht']), 'chi_S': chi_S, 'acc': res['acc'],
            }
            results.append(r)

            marker = ''
            if r['H'] > 0.6 and beta > 0.5:
                marker = ' *'

            print(f"  {beta:>7.2f} {r['S']:>8.2f} {r['H']:>7.3f} {r['H_std']:>7.3f} "
                  f"{r['f']:>6.3f} {r['ht']:>5.1f} {chi_S:>8.1f} {r['acc']:>5.3f}{marker}",
                  flush=True)

        # Find transition and non-monotonic features
        H_vals = [r['H'] for r in results]
        chi_vals = [r['chi_S'] for r in results]
        f_vals = [r['f'] for r in results]

        chi_peak = np.argmax(chi_vals)
        H_peak = np.argmax(H_vals)

        print(f"\n  chi_S peak at beta = {results[chi_peak]['beta']:.2f} "
              f"(chi = {chi_vals[chi_peak]:.1f})")
        print(f"  H peak at beta = {results[H_peak]['beta']:.2f} "
              f"(H = {H_vals[H_peak]:.3f})")

        # Check for non-monotonicity: does H dip below its initial value then recover?
        H_init = H_vals[0]
        H_min_post_peak = min(H_vals[H_peak:]) if H_peak < len(H_vals) - 1 else H_vals[-1]
        H_final = H_vals[-1]

        print(f"  H(beta=0) = {H_init:.3f}")
        print(f"  H(peak) = {H_vals[H_peak]:.3f}")
        print(f"  H(min after peak) = {H_min_post_peak:.3f}")
        print(f"  H(max beta) = {H_final:.3f}")

        if H_vals[H_peak] > H_init + 0.05 and H_min_post_peak < H_vals[H_peak] - 0.1:
            print(f"  --> NON-MONOTONIC: H rises then dips")
            if H_final > H_min_post_peak + 0.1:
                print(f"  --> AND RECOVERS: possible intermediate phase")
        else:
            print(f"  --> Roughly monotonic")

        elapsed = time.time() - t_start
        print(f"  [{elapsed:.0f}s total]")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()

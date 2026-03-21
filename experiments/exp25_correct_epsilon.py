"""
Experiment 25: BD Phase Transition at Correct Epsilon

Key insight from exp24: Surya uses epsilon ≈ 0.05 (NOT epsilon=1).
At epsilon=0.05, N=50: beta_c = 1.66 / (50 * 0.0025) = 13.3

Previous experiments used epsilon=1, which puts beta_c ≈ 0.033 — but
the action formula at epsilon=1 has a singularity in f_2(n, eps)
that makes the sum behave differently.

This experiment: run at epsilon=0.05 with beta scanning through beta_c ≈ 13.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
import time
from causal_sets.two_orders import TwoOrder, swap_move
from causal_sets.bd_action import count_intervals_by_size
from causal_sets.fast_core import FastCausalSet, spectral_dimension_fast
from causal_sets.core import CausalSet
from causal_sets.dimension import myrheim_meyer


def bd_action_epsilon(cs: FastCausalSet, eps: float) -> float:
    """Glaser's action: S = 4*eps*(N - 2*eps*sum_n N_n * f_2(n, eps))"""
    N = cs.n
    max_k = min(N - 2, 20)
    counts = count_intervals_by_size(cs, max_size=max_k)

    total = 0.0
    for n in range(max_k + 1):
        if n not in counts or counts[n] == 0:
            continue
        if abs(1 - eps) < 1e-10:
            f2 = 1.0 if n == 0 else 0.0
        else:
            r = (1 - eps) ** n
            f2 = r * (1 - 2 * eps * n / (1 - eps) +
                       eps ** 2 * n * (n - 1) / (2 * (1 - eps) ** 2))
        total += counts[n] * f2

    return 4 * eps * (N - 2 * eps * total)


def mcmc_epsilon(N, beta, eps, n_steps=50000, n_therm=25000,
                  record_every=20, rng=None):
    """MCMC on 2-orders with the epsilon-dependent action."""
    if rng is None:
        rng = np.random.default_rng()

    current = TwoOrder(N, rng=rng)
    current_cs = current.to_causet()
    current_S = bd_action_epsilon(current_cs, eps)

    samples = []
    actions = []
    n_acc = 0

    for step in range(n_steps):
        proposed = swap_move(current, rng)
        proposed_cs = proposed.to_causet()
        proposed_S = bd_action_epsilon(proposed_cs, eps)

        dS = beta * (proposed_S - current_S)
        if dS <= 0 or rng.random() < np.exp(-dS):
            current = proposed
            current_cs = proposed_cs
            current_S = proposed_S
            n_acc += 1

        if step >= n_therm and step % record_every == 0:
            actions.append(current_S)
            samples.append(current_cs)

    return {
        'actions': np.array(actions),
        'samples': samples,
        'accept_rate': n_acc / n_steps,
    }


def measure(cs, compute_spectral=True):
    n = cs.n
    of = cs.ordering_fraction()
    chain = cs.longest_chain()
    links = int(np.sum(cs.link_matrix()))

    cs_old = CausalSet(n)
    cs_old.order = cs.order.astype(np.int8)
    mm = myrheim_meyer(cs_old)

    ds_peak = float('nan')
    if compute_spectral and n >= 15:
        sigmas, d_s = spectral_dimension_fast(cs, sigma_range=(0.1, 50.0), n_sigma=30)
        if len(d_s) > 0:
            ds_peak = np.max(d_s)

    return {'of': of, 'chain': chain, 'links_n': links / n, 'mm': mm, 'ds_peak': ds_peak}


def main():
    rng = np.random.default_rng(42)
    N = 50
    eps = 0.05

    beta_c_pred = 1.66 / (N * eps ** 2)

    print("=" * 80)
    print(f"EXPERIMENT 25: BD Phase Transition at eps={eps}")
    print(f"N={N}, predicted beta_c = {beta_c_pred:.1f}")
    print("=" * 80)

    # Phase 1: Verify action at beta=0
    print(f"\n--- Verify: <S> at beta=0, eps={eps} ---")
    res = mcmc_epsilon(N, beta=0.0, eps=eps, n_steps=20000, n_therm=10000, rng=rng)
    print(f"  <S> = {np.mean(res['actions']):.2f} ± {np.std(res['actions']):.2f}")
    print(f"  <S/N> = {np.mean(res['actions'])/N:.3f}")
    print(f"  Surya reports S ≈ 4 at beta=0, N=50")

    # Phase 2: Scan beta through the transition
    print(f"\n--- Beta scan through beta_c ≈ {beta_c_pred:.1f} ---")
    betas = np.array([0, 2, 5, 8, 10, 12, 13, 14, 15, 17, 20, 25, 30, 40, 60])

    print(f"  {'beta':>6} {'<S>':>8} {'<S/N>':>8} {'f':>6} {'h':>5} {'L/N':>6} "
          f"{'d_s':>6} {'acc':>6} {'chi':>8}")
    print("-" * 75)

    all_results = []
    for beta in betas:
        t0 = time.time()
        res = mcmc_epsilon(N, beta=beta, eps=eps,
                            n_steps=40000, n_therm=20000, record_every=20, rng=rng)

        obs = [measure(cs, compute_spectral=(i % 30 == 0)) for i, cs in enumerate(res['samples'])]

        S_mean = np.mean(res['actions'])
        S_std = np.std(res['actions'])
        chi = np.var(res['actions']) * N
        f = np.mean([o['of'] for o in obs])
        h = np.mean([o['chain'] for o in obs])
        ln = np.mean([o['links_n'] for o in obs])
        ds_vals = [o['ds_peak'] for o in obs if not np.isnan(o['ds_peak'])]
        ds = np.mean(ds_vals) if ds_vals else float('nan')

        marker = ""
        if f > 0.55:
            marker = " <-- CRYSTALLINE"

        elapsed = time.time() - t0

        print(f"  {beta:>6.0f} {S_mean:>8.2f} {S_mean/N:>8.3f} {f:>6.3f} {h:>5.1f} "
              f"{ln:>6.2f} {ds:>6.2f} {res['accept_rate']:>6.3f} {chi:>8.1f}{marker}")

        all_results.append({
            'beta': beta, 'S_mean': S_mean, 'S_std': S_std, 'chi': chi,
            'f': f, 'h': h, 'ln': ln, 'ds': ds, 'accept': res['accept_rate']
        })

    # Find transition
    chis = [r['chi'] for r in all_results]
    peak = np.argmax(chis)
    beta_c_meas = all_results[peak]['beta']

    print(f"\n  Susceptibility peak at beta = {beta_c_meas:.0f}")
    print(f"  Predicted beta_c = {beta_c_pred:.1f}")
    print(f"  Agreement: {abs(beta_c_meas - beta_c_pred) / beta_c_pred * 100:.0f}%")

    # Spectral dimension in each phase
    pre = [r for r in all_results if r['beta'] < beta_c_meas * 0.5]
    post = [r for r in all_results if r['beta'] > beta_c_meas * 2]

    if pre:
        ds_pre = [r['ds'] for r in pre if not np.isnan(r['ds'])]
        f_pre = [r['f'] for r in pre]
        h_pre = [r['h'] for r in pre]
        print(f"\n  CONTINUUM PHASE (beta < {beta_c_meas*0.5:.0f}):")
        print(f"    d_s = {np.mean(ds_pre):.2f}, f = {np.mean(f_pre):.3f}, h = {np.mean(h_pre):.1f}")

    if post:
        ds_post = [r['ds'] for r in post if not np.isnan(r['ds'])]
        f_post = [r['f'] for r in post]
        h_post = [r['h'] for r in post]
        print(f"  CRYSTALLINE PHASE (beta > {beta_c_meas*2:.0f}):")
        print(f"    d_s = {np.mean(ds_post):.2f}, f = {np.mean(f_post):.3f}, h = {np.mean(h_post):.1f}")

    if pre and post and ds_pre and ds_post:
        delta = np.mean(ds_post) - np.mean(ds_pre)
        print(f"\n  SPECTRAL DIMENSION CHANGE: Delta_d_s = {delta:+.2f}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()

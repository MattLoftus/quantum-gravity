"""
Experiment 52: Large-N SJ Vacuum on MCMC-Sampled Continuum-Phase Causets

Exp49 showed that RANDOM 2-orders have c_eff diverging (3→4.1) and ER=EPR
gap vanishing at large N. But random 2-orders are NOT the same as
continuum-phase causets. The BD partition function at low beta selects
manifold-like causets.

Key question: On MCMC-sampled continuum-phase 2-orders, does c_eff → 1?

We run parallel tempering MCMC at N=50, 100, 150, 200, sample configurations
from the continuum phase (beta slightly below beta_c), then compute:
  (a) c_eff = 3*S(N/2)/ln(N)
  (b) ER=EPR correlation r
  (c) Level spacing ratio <r>
  (d) Random DAG control for ER=EPR
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
import time
from causal_sets.two_orders import TwoOrder, swap_move
from causal_sets.two_orders_v2 import bd_action_corrected
from causal_sets.fast_core import FastCausalSet

rng = np.random.default_rng(42)


def sj_vacuum(cs):
    """Compute SJ Wightman function."""
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
    if len(region) < 2:
        return 0.0
    W_A = W[np.ix_(region, region)]
    eigs = np.linalg.eigvalsh(W_A)
    eigs = np.clip(eigs, 1e-15, 1 - 1e-15)
    return float(-np.sum(eigs * np.log(eigs) + (1 - eigs) * np.log(1 - eigs)))


def er_epr_correlation(cs, W):
    """Compute correlation between |W[i,j]| and connectivity for spacelike pairs."""
    N = cs.n
    order_int = cs.order.astype(np.int32)
    w_vals = []
    kappa_vals = []
    for i in range(N):
        for j in range(i + 1, N):
            if not cs.order[i, j] and not cs.order[j, i]:
                kappa = int(np.sum(order_int[:, i] & order_int[:, j])) + \
                        int(np.sum(order_int[i, :] & order_int[j, :]))
                w_vals.append(abs(W[i, j]))
                kappa_vals.append(kappa)
    if len(w_vals) < 10:
        return 0.0
    return float(np.corrcoef(w_vals, kappa_vals)[0, 1])


def level_spacing_ratio(evals):
    """Average level spacing ratio for positive eigenvalues."""
    pos = sorted(evals[evals > 1e-12])
    if len(pos) < 4:
        return 0.0
    spacings = np.diff(pos)
    ratios = []
    for i in range(len(spacings) - 1):
        s1, s2 = spacings[i], spacings[i + 1]
        if max(s1, s2) > 1e-15:
            ratios.append(min(s1, s2) / max(s1, s2))
    return float(np.mean(ratios)) if ratios else 0.0


def parallel_tempering_mcmc(N, beta_target, eps, n_steps=30000, n_therm=15000,
                             n_chains=6, swap_interval=15, record_every=50, rng=None):
    """
    Parallel tempering MCMC returning TwoOrder samples from the target beta chain.
    Uses a geometric beta ladder from 0 to beta_target.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Beta ladder: geometric from small to target
    betas = np.array([beta_target * (i / (n_chains - 1))**2 for i in range(n_chains)])
    betas[0] = 0.0  # coldest chain is free

    # Initialize chains
    chains = [TwoOrder(N, rng=rng) for _ in range(n_chains)]
    chain_cs = [c.to_causet() for c in chains]
    chain_S = [bd_action_corrected(cs, eps) for cs in chain_cs]

    samples = []
    actions = []
    n_acc = [0] * n_chains
    n_swaps = 0
    n_swap_acc = 0

    for step in range(n_steps):
        # MCMC move on each chain
        for ci in range(n_chains):
            proposed = swap_move(chains[ci], rng)
            proposed_cs = proposed.to_causet()
            proposed_S = bd_action_corrected(proposed_cs, eps)

            dS = betas[ci] * (proposed_S - chain_S[ci])
            if dS <= 0 or rng.random() < np.exp(-min(dS, 500)):
                chains[ci] = proposed
                chain_cs[ci] = proposed_cs
                chain_S[ci] = proposed_S
                n_acc[ci] += 1

        # Swap adjacent chains
        if step % swap_interval == 0 and step > 0:
            for ci in range(n_chains - 1):
                n_swaps += 1
                dS = (betas[ci + 1] - betas[ci]) * (chain_S[ci] - chain_S[ci + 1])
                if dS <= 0 or rng.random() < np.exp(-min(dS, 500)):
                    chains[ci], chains[ci + 1] = chains[ci + 1], chains[ci]
                    chain_cs[ci], chain_cs[ci + 1] = chain_cs[ci + 1], chain_cs[ci]
                    chain_S[ci], chain_S[ci + 1] = chain_S[ci + 1], chain_S[ci]
                    n_swap_acc += 1

        # Record from highest-beta chain (target)
        if step >= n_therm and step % record_every == 0:
            target_idx = n_chains - 1
            samples.append(chains[target_idx].copy())
            actions.append(chain_S[target_idx])

    acc_rates = [n / n_steps for n in n_acc]
    swap_rate = n_swap_acc / max(n_swaps, 1)

    return {
        'samples': samples,
        'actions': np.array(actions),
        'acc_rates': acc_rates,
        'swap_rate': swap_rate,
        'betas': betas,
    }


print("=" * 80)
print("EXPERIMENT 52: MCMC-SAMPLED CONTINUUM-PHASE CAUSETS AT LARGE N")
print("=" * 80)
print()

eps = 0.12
results = []

for N in [50, 100, 150, 200]:
    beta_c = 1.66 / (N * eps**2)
    # Target: continuum phase, slightly below beta_c
    beta_target = 0.7 * beta_c

    print(f"--- N = {N}, beta_c = {beta_c:.2f}, beta_target = {beta_target:.2f} ---")
    t0 = time.time()

    # Adjust MCMC parameters for larger N
    n_steps = max(20000, int(40000 * (50 / N)))
    n_therm = n_steps // 2
    n_chains = 6
    record_every = max(30, n_steps // 200)

    print(f"  MCMC: {n_steps} steps, {n_therm} thermalization, record every {record_every}")

    mcmc = parallel_tempering_mcmc(
        N, beta_target, eps,
        n_steps=n_steps, n_therm=n_therm, n_chains=n_chains,
        swap_interval=15, record_every=record_every, rng=rng
    )

    t_mcmc = time.time() - t0
    n_samples = len(mcmc['samples'])
    print(f"  MCMC done: {t_mcmc:.0f}s, {n_samples} samples, "
          f"acc={mcmc['acc_rates'][-1]:.1%} (target), swap={mcmc['swap_rate']:.1%}")

    if n_samples == 0:
        print(f"  WARNING: No samples collected!")
        continue

    # Use last few samples for SJ vacuum analysis
    n_analyze = min(3, n_samples)
    c_effs = []
    r_ers = []
    r_dags = []
    lsrs = []

    for si in range(n_analyze):
        sample = mcmc['samples'][-(si + 1)]
        cs = sample.to_causet()

        # Ordering fraction (manifold-likeness check)
        f = np.sum(cs.order) / (N * (N - 1))

        t1 = time.time()
        W, evals = sj_vacuum(cs)
        t_sj = time.time() - t1

        # (a) Central charge
        A = list(range(N // 2))
        S_half = entanglement_entropy(W, A)
        c_eff = 3 * S_half / np.log(N)
        c_effs.append(c_eff)

        # (b) ER=EPR
        r_er = er_epr_correlation(cs, W)
        r_ers.append(r_er)

        # (c) Level spacing ratio
        lsr = level_spacing_ratio(evals)
        lsrs.append(lsr)

        # (d) Random DAG control
        order_rand = np.zeros((N, N), dtype=bool)
        perm = rng.permutation(N)
        target_density = np.sum(cs.order) / (N * (N - 1))
        for i in range(N):
            for j in range(i + 1, N):
                if rng.random() < target_density:
                    order_rand[perm[i], perm[j]] = True
        cs_r = FastCausalSet(N)
        cs_r.order = order_rand
        W_r, _ = sj_vacuum(cs_r)
        r_dag = er_epr_correlation(cs_r, W_r)
        r_dags.append(r_dag)

        if si == 0:
            print(f"  Sample {si}: f={f:.3f}, S(N/2)={S_half:.3f}, c_eff={c_eff:.3f}, "
                  f"r_ER={r_er:.3f}, r_DAG={r_dag:.3f}, <r>={lsr:.4f}, SJ={t_sj:.1f}s")

    result = {
        'N': N,
        'beta': beta_target,
        'beta_c': beta_c,
        'c_eff': np.mean(c_effs),
        'c_eff_std': np.std(c_effs),
        'r_er': np.mean(r_ers),
        'r_er_std': np.std(r_ers),
        'r_dag': np.mean(r_dags),
        'r_dag_std': np.std(r_dags),
        'lsr': np.mean(lsrs),
        'lsr_std': np.std(lsrs),
        'time': time.time() - t0,
        'n_samples': n_samples,
        'acc_rate': mcmc['acc_rates'][-1],
        'swap_rate': mcmc['swap_rate'],
    }
    results.append(result)

    print(f"  Summary: c_eff={result['c_eff']:.3f}±{result['c_eff_std']:.3f}, "
          f"r_ER={result['r_er']:.3f}±{result['r_er_std']:.3f}, "
          f"gap={result['r_er']-result['r_dag']:.3f}, "
          f"<r>={result['lsr']:.4f}")
    print(f"  Total time: {result['time']:.0f}s")
    print()


# ================================================================
# Also run random 2-orders at same N for comparison
# ================================================================
print("=" * 80)
print("COMPARISON: RANDOM 2-ORDERS (same N values)")
print("=" * 80)
print()

random_results = []
for N in [50, 100, 150, 200]:
    c_effs_r = []
    for trial in range(3):
        to = TwoOrder(N, rng=rng)
        cs = to.to_causet()
        W, evals = sj_vacuum(cs)
        S_half = entanglement_entropy(W, list(range(N // 2)))
        c_effs_r.append(3 * S_half / np.log(N))

    print(f"N={N}: c_eff(random) = {np.mean(c_effs_r):.3f} ± {np.std(c_effs_r):.3f}")
    random_results.append({'N': N, 'c_eff': np.mean(c_effs_r), 'c_eff_std': np.std(c_effs_r)})


# ================================================================
# SUMMARY
# ================================================================
print()
print("=" * 80)
print("SUMMARY: MCMC CONTINUUM PHASE vs RANDOM 2-ORDERS")
print("=" * 80)
print()
print(f"{'N':>5} | {'c_eff(MCMC)':>12} | {'c_eff(random)':>14} | {'ratio':>6} | {'r_ER':>6} | {'gap':>6} | {'<r>':>6}")
print("-" * 75)
for r, rr in zip(results, random_results):
    ratio = r['c_eff'] / rr['c_eff'] if rr['c_eff'] > 0 else 0
    gap = r['r_er'] - r['r_dag']
    print(f"{r['N']:>5} | {r['c_eff']:>8.3f}±{r['c_eff_std']:.3f} | "
          f"{rr['c_eff']:>10.3f}±{rr['c_eff_std']:.3f} | {ratio:>6.3f} | "
          f"{r['r_er']:>6.3f} | {gap:>+6.3f} | {r['lsr']:>6.4f}")

print()
print("KEY QUESTION: Does c_eff(MCMC) < c_eff(random)?")
print("If yes → BD action selects causets closer to CFT (c=1)")
print("If no → the c divergence is intrinsic to the SJ construction on 2-orders")
print()
print("GUE reference: <r> = 0.5996")
print("Poisson reference: <r> = 0.3863")

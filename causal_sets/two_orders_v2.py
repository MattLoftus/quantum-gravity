"""
Corrected 2-order BD action and MCMC.

The action is: S/hbar = eps * (N - 2*eps * sum_n N_n * f(n, eps))
NOT 4*eps*(...) — the factor of 4 was an error in our implementation.

With this correction:
  - At eps=0.12, N=50, beta=0: <S> ≈ 3.5 (Surya reports 3.846)
  - beta_c = 1.66 / (N * eps^2) from Glaser et al.
"""

import numpy as np
from .two_orders import TwoOrder, swap_move
from .bd_action import count_intervals_by_size
from .fast_core import FastCausalSet


def bd_action_corrected(cs: FastCausalSet, eps: float) -> float:
    """Corrected BD action: S = eps*(N - 2*eps*sum N_n*f(n,eps))"""
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
    return eps * (N - 2 * eps * total)


def mcmc_corrected(N, beta, eps, n_steps=50000, n_therm=25000,
                    record_every=20, rng=None):
    """MCMC with corrected action."""
    if rng is None:
        rng = np.random.default_rng()

    current = TwoOrder(N, rng=rng)
    current_cs = current.to_causet()
    current_S = bd_action_corrected(current_cs, eps)

    samples = []
    actions = []
    n_acc = 0

    for step in range(n_steps):
        proposed = swap_move(current, rng)
        proposed_cs = proposed.to_causet()
        proposed_S = bd_action_corrected(proposed_cs, eps)

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


def parallel_tempering(N, betas, eps, n_steps=50000, swap_interval=20,
                        n_therm=25000, record_every=25, rng=None):
    """
    Parallel tempering MCMC for 2-orders.

    Runs independent MCMC chains at each beta value. Every `swap_interval`
    steps, proposes swapping configurations between adjacent-beta chains.
    Swap accepted with probability min(1, exp((beta_i - beta_j)*(S_i - S_j))).

    This allows high-beta (crystalline) chains to escape local minima via
    exchanges with lower-beta chains that explore more freely.

    Parameters
    ----------
    N : int
        Number of causal set elements.
    betas : array-like
        Temperature parameters, must be sorted ascending.
    eps : float
        Non-locality parameter for BD action.
    n_steps : int
        Total MCMC steps per chain.
    swap_interval : int
        Attempt swap between adjacent chains every this many steps.
    n_therm : int
        Thermalization steps (samples discarded).
    record_every : int
        Record samples every this many steps after thermalization.
    rng : np.random.Generator or None

    Returns
    -------
    dict with keys:
        'chains': list of dicts, one per beta, each with 'actions', 'samples',
                  'accept_rate', 'swap_rate'
        'betas': the beta array used
        'swap_matrix': (n_chains-1,) array of swap acceptance rates
    """
    if rng is None:
        rng = np.random.default_rng()

    betas = np.sort(np.asarray(betas, dtype=float))
    n_chains = len(betas)

    # Initialize chains
    configs = [TwoOrder(N, rng=rng) for _ in range(n_chains)]
    causets = [c.to_causet() for c in configs]
    actions = [bd_action_corrected(cs, eps) for cs in causets]

    # Tracking
    chain_samples = [[] for _ in range(n_chains)]
    chain_actions = [[] for _ in range(n_chains)]
    chain_accepts = [0 for _ in range(n_chains)]
    swap_attempts = np.zeros(n_chains - 1, dtype=int)
    swap_accepts = np.zeros(n_chains - 1, dtype=int)

    for step in range(n_steps):
        # Standard MCMC move on each chain
        for c in range(n_chains):
            proposed = swap_move(configs[c], rng)
            proposed_cs = proposed.to_causet()
            proposed_S = bd_action_corrected(proposed_cs, eps)

            dS = betas[c] * (proposed_S - actions[c])
            if dS <= 0 or rng.random() < np.exp(-dS):
                configs[c] = proposed
                causets[c] = proposed_cs
                actions[c] = proposed_S
                chain_accepts[c] += 1

        # Swap proposals between adjacent chains
        if step > 0 and step % swap_interval == 0:
            # Randomly choose even or odd pairs to avoid bias
            if rng.random() < 0.5:
                pairs = range(0, n_chains - 1, 2)
            else:
                pairs = range(1, n_chains - 1, 2)

            for i in pairs:
                j = i + 1
                swap_attempts[i] += 1
                # Metropolis criterion for replica exchange
                delta = (betas[i] - betas[j]) * (actions[i] - actions[j])
                if delta >= 0 or rng.random() < np.exp(delta):
                    swap_accepts[i] += 1
                    # Swap configurations (not betas)
                    configs[i], configs[j] = configs[j], configs[i]
                    causets[i], causets[j] = causets[j], causets[i]
                    actions[i], actions[j] = actions[j], actions[i]

        # Record samples after thermalization
        if step >= n_therm and step % record_every == 0:
            for c in range(n_chains):
                chain_actions[c].append(actions[c])
                chain_samples[c].append(causets[c])

    # Build results
    chains = []
    for c in range(n_chains):
        chains.append({
            'actions': np.array(chain_actions[c]),
            'samples': chain_samples[c],
            'accept_rate': chain_accepts[c] / n_steps,
            'beta': betas[c],
        })

    swap_rates = np.where(swap_attempts > 0,
                          swap_accepts / swap_attempts, 0.0)

    return {
        'chains': chains,
        'betas': betas,
        'swap_matrix': swap_rates,
    }

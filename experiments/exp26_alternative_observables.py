"""
Experiment 26: Alternative Observables Across the BD Phase Transition

Link-graph spectral dimension is purely a function of link density (exp25).
We need observables that capture CAUSAL STRUCTURE, not just connectivity.

Four candidates:
1. Chain-based dimension: longest chain ~ N^(1/d), independent of link density
2. Interval abundance distribution: how N_k values distribute across k
3. Curvature via BD action density: local action variations
4. Homology: topological features (connected components, cycles)

We measure all four across the BD phase transition (eps=0.05, N=50)
to find which ones actually distinguish the continuum from crystalline phase.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
import time
from causal_sets.two_orders import TwoOrder, swap_move
from causal_sets.bd_action import count_intervals_by_size
from causal_sets.fast_core import FastCausalSet
from causal_sets.core import CausalSet
from causal_sets.dimension import myrheim_meyer


def bd_action_eps(cs, eps):
    """Compute BD action at given epsilon."""
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


def mcmc_eps(N, beta, eps, n_steps=40000, n_therm=20000, record_every=20, rng=None):
    """MCMC on 2-orders."""
    if rng is None:
        rng = np.random.default_rng()
    current = TwoOrder(N, rng=rng)
    current_cs = current.to_causet()
    current_S = bd_action_eps(current_cs, eps)
    samples = []
    actions = []
    n_acc = 0
    for step in range(n_steps):
        proposed = swap_move(current, rng)
        proposed_cs = proposed.to_causet()
        proposed_S = bd_action_eps(proposed_cs, eps)
        dS = beta * (proposed_S - current_S)
        if dS <= 0 or rng.random() < np.exp(-dS):
            current = proposed
            current_cs = proposed_cs
            current_S = proposed_S
            n_acc += 1
        if step >= n_therm and step % record_every == 0:
            actions.append(current_S)
            samples.append(current_cs)
    return {'actions': np.array(actions), 'samples': samples,
            'accept_rate': n_acc / n_steps}


# ============================================================
# Observable 1: Chain-based dimension estimator
# ============================================================
def chain_dimension(cs):
    """
    Estimate dimension from longest chain scaling.
    For a d-dimensional manifold with N elements: longest_chain ~ N^(1/d)
    So d_chain = log(N) / log(longest_chain)
    """
    N = cs.n
    chain = cs.longest_chain()
    if chain <= 1:
        return float('nan')
    return np.log(N) / np.log(chain)


def chain_profile(cs):
    """Distribution of chain lengths from each element."""
    N = cs.n
    # Forward chain length from each element
    dp = np.ones(N, dtype=int)
    order = cs.order
    for i in range(N - 2, -1, -1):
        successors = np.where(order[i, i + 1:])[0] + i + 1
        if len(successors) > 0:
            dp[i] = 1 + np.max(dp[successors])
    return dp


# ============================================================
# Observable 2: Interval abundance distribution
# ============================================================
def interval_distribution(cs, max_k=15):
    """
    Compute the full interval abundance distribution N_k for k=0..max_k.
    The SHAPE of this distribution encodes geometric information
    that link density alone cannot capture.
    """
    counts = count_intervals_by_size(cs, max_size=max_k)
    dist = np.zeros(max_k + 1)
    for k in range(max_k + 1):
        dist[k] = counts.get(k, 0)
    return dist


def interval_entropy(dist):
    """Shannon entropy of the normalized interval distribution.
    High entropy = many different interval sizes (manifold-like).
    Low entropy = dominated by one size (non-manifold)."""
    p = dist / (np.sum(dist) + 1e-30)
    p = p[p > 0]
    return -np.sum(p * np.log(p + 1e-30))


def interval_concentration(dist):
    """Fraction of intervals that are links (k=0).
    High = most relations are direct (tree-like).
    Low = many indirect relations (manifold-like)."""
    total = np.sum(dist)
    if total == 0:
        return float('nan')
    return dist[0] / total


# ============================================================
# Observable 3: Local curvature via BD action
# ============================================================
def local_action_variance(cs, eps):
    """
    Compute the per-element BD action contribution and its variance.
    The variance measures curvature fluctuations.
    Flat space → low variance. Curved/non-manifold → high variance.
    """
    N = cs.n
    order_int = cs.order.astype(np.int32)
    interval_matrix = order_int @ order_int

    # For each element x, count its contribution to intervals of each size
    # Element x contributes to interval [a,b] if a < x < b
    # The per-element action contribution is approximately:
    # s(x) = 1 - 2*eps*(links from x) + corrections

    # Simpler: count links per element
    links_matrix = cs.link_matrix()
    links_out = np.sum(links_matrix, axis=1)  # outgoing links
    links_in = np.sum(links_matrix, axis=0)   # incoming links
    links_total = links_out + links_in  # total links per element

    return {
        'mean_links': np.mean(links_total),
        'std_links': np.std(links_total),
        'cv_links': np.std(links_total) / (np.mean(links_total) + 1e-10),
        'max_links': int(np.max(links_total)),
        'min_links': int(np.min(links_total)),
    }


# ============================================================
# Observable 4: Topological / homological features
# ============================================================
def causal_topology(cs):
    """
    Compute topological features of the causal set.
    - Number of maximal elements (future boundary)
    - Number of minimal elements (past boundary)
    - Width (maximum antichain size)
    - Layered structure indicator
    """
    N = cs.n
    order = cs.order

    # Maximal elements: no successors
    n_maximal = 0
    for i in range(N):
        if not np.any(order[i, :]):
            n_maximal += 1

    # Minimal elements: no predecessors
    n_minimal = 0
    for i in range(N):
        if not np.any(order[:, i]):
            n_minimal += 1

    # Width estimate: length of longest antichain
    # Exact computation is NP-hard; use Dilworth's theorem:
    # width = N - longest_chain (approximate for partial orders)
    chain = cs.longest_chain()
    width_estimate = N - chain  # rough upper bound

    # Layer structure: compute "height" of each element
    heights = np.zeros(N, dtype=int)
    for j in range(N):
        predecessors = np.where(order[:, j])[0]
        if len(predecessors) > 0:
            heights[j] = np.max(heights[predecessors]) + 1

    n_layers = len(np.unique(heights))
    layer_sizes = [np.sum(heights == h) for h in range(int(np.max(heights)) + 1)]
    layer_uniformity = np.std(layer_sizes) / (np.mean(layer_sizes) + 1e-10)

    return {
        'n_maximal': n_maximal,
        'n_minimal': n_minimal,
        'width_est': width_estimate,
        'n_layers': n_layers,
        'layer_uniformity': layer_uniformity,
        'max_layer': int(np.max(layer_sizes)) if layer_sizes else 0,
    }


def main():
    rng = np.random.default_rng(42)
    N = 50
    eps = 0.05

    print("=" * 90)
    print("EXPERIMENT 26: Alternative Observables Across BD Phase Transition")
    print(f"N={N}, eps={eps}, beta_c ≈ {1.66/(N*eps**2):.1f}")
    print("=" * 90)

    betas = [0, 5, 10, 15, 17, 20, 25, 30, 40, 60]

    # Headers
    print(f"\n{'beta':>6} | {'CHAIN':^25} | {'INTERVALS':^30} | {'CURVATURE':^20} | {'TOPOLOGY':^30}")
    print(f"{'':>6} | {'d_chain':>7} {'height':>7} {'h/N':>7} | "
          f"{'entropy':>8} {'link_frac':>9} {'N1/N0':>7} | "
          f"{'cv_links':>9} {'std_lnk':>8} | "
          f"{'n_max':>6} {'n_min':>6} {'layers':>7} {'lyr_cv':>7}")
    print("-" * 140)

    all_results = []

    for beta in betas:
        t0 = time.time()
        res = mcmc_eps(N, beta=beta, eps=eps, n_steps=40000, n_therm=20000,
                        record_every=20, rng=rng)

        # Measure all observables on samples
        chain_dims = []
        heights = []
        int_entropies = []
        link_fracs = []
        n1_n0_ratios = []
        cv_links_list = []
        std_links_list = []
        n_maximal_list = []
        n_minimal_list = []
        n_layers_list = []
        layer_cv_list = []

        for cs in res['samples']:
            # 1. Chain dimension
            cd = chain_dimension(cs)
            if not np.isnan(cd):
                chain_dims.append(cd)
            heights.append(cs.longest_chain())

            # 2. Interval distribution
            dist = interval_distribution(cs, max_k=15)
            int_entropies.append(interval_entropy(dist))
            link_fracs.append(interval_concentration(dist))
            n1_n0 = dist[1] / (dist[0] + 1e-10)
            n1_n0_ratios.append(n1_n0)

            # 3. Local curvature
            curv = local_action_variance(cs, eps)
            cv_links_list.append(curv['cv_links'])
            std_links_list.append(curv['std_links'])

            # 4. Topology
            topo = causal_topology(cs)
            n_maximal_list.append(topo['n_maximal'])
            n_minimal_list.append(topo['n_minimal'])
            n_layers_list.append(topo['n_layers'])
            layer_cv_list.append(topo['layer_uniformity'])

        elapsed = time.time() - t0

        d_chain = np.mean(chain_dims) if chain_dims else float('nan')
        h = np.mean(heights)

        print(f"{beta:>6.0f} | {d_chain:>7.2f} {h:>7.1f} {h/N:>7.3f} | "
              f"{np.mean(int_entropies):>8.3f} {np.mean(link_fracs):>9.3f} "
              f"{np.mean(n1_n0_ratios):>7.3f} | "
              f"{np.mean(cv_links_list):>9.3f} {np.mean(std_links_list):>8.2f} | "
              f"{np.mean(n_maximal_list):>6.1f} {np.mean(n_minimal_list):>6.1f} "
              f"{np.mean(n_layers_list):>7.1f} {np.mean(layer_cv_list):>7.3f}  "
              f"({elapsed:.0f}s)")

        all_results.append({
            'beta': beta,
            'd_chain': d_chain, 'height': h,
            'int_entropy': np.mean(int_entropies),
            'link_frac': np.mean(link_fracs),
            'n1_n0': np.mean(n1_n0_ratios),
            'cv_links': np.mean(cv_links_list),
            'n_maximal': np.mean(n_maximal_list),
            'n_minimal': np.mean(n_minimal_list),
            'n_layers': np.mean(n_layers_list),
            'layer_cv': np.mean(layer_cv_list),
            'accept': res['accept_rate'],
        })

    # Analysis: which observables show a clear transition?
    print("\n" + "=" * 90)
    print("ANALYSIS: Which observables distinguish the phases?")
    print("=" * 90)

    # Split into continuum (beta < 10) and crystalline (beta > 25)
    cont = [r for r in all_results if r['beta'] < 10]
    cryst = [r for r in all_results if r['beta'] > 25]

    if cont and cryst:
        print(f"\n{'Observable':>25} {'Continuum':>12} {'Crystalline':>12} {'Change':>10} {'Signal?':>10}")
        print("-" * 75)

        for obs_name in ['d_chain', 'height', 'int_entropy', 'link_frac', 'n1_n0',
                          'cv_links', 'n_maximal', 'n_minimal', 'n_layers', 'layer_cv']:
            c_val = np.mean([r[obs_name] for r in cont if not np.isnan(r[obs_name])])
            x_val = np.mean([r[obs_name] for r in cryst if not np.isnan(r[obs_name])])

            if abs(c_val) > 0.001:
                change = (x_val - c_val) / abs(c_val) * 100
                signal = "YES" if abs(change) > 20 else "maybe" if abs(change) > 10 else "no"
            else:
                change = float('nan')
                signal = "n/a"

            print(f"  {obs_name:>23} {c_val:>12.3f} {x_val:>12.3f} {change:>+9.1f}% {signal:>10}")

    # Compare with random graph control
    print(f"\n--- Random graph control (same link density as crystalline phase) ---")
    # Get crystalline L/N
    if cryst:
        cryst_samples = []
        res_cryst = mcmc_eps(N, beta=60, eps=eps, n_steps=30000, n_therm=20000,
                              record_every=100, rng=rng)
        for cs in res_cryst['samples'][-10:]:
            links = int(np.sum(cs.link_matrix()))
            ln = links / N

            # Random graph with same L/N
            p = 2 * ln / (N - 1)
            p = min(p, 0.99)

            adj = np.zeros((N, N), dtype=bool)
            for i in range(N):
                for j in range(i + 1, N):
                    if rng.random() < p:
                        adj[i, j] = True

            # Make it a DAG (upper triangular = valid partial order)
            rand_cs = FastCausalSet(N)
            rand_cs.order = adj

            cd_causet = chain_dimension(cs)
            cd_random = chain_dimension(rand_cs)

            dist_c = interval_distribution(cs)
            dist_r = interval_distribution(rand_cs)

            topo_c = causal_topology(cs)
            topo_r = causal_topology(rand_cs)

            cryst_samples.append({
                'causet': {'d_chain': cd_causet, 'int_entropy': interval_entropy(dist_c),
                           'n_layers': topo_c['n_layers'], 'layer_cv': topo_c['layer_uniformity']},
                'random': {'d_chain': cd_random, 'int_entropy': interval_entropy(dist_r),
                           'n_layers': topo_r['n_layers'], 'layer_cv': topo_r['layer_uniformity']},
            })

        print(f"\n  {'Observable':>20} {'BD Causet':>12} {'Random DAG':>12} {'Different?':>12}")
        print("  " + "-" * 60)
        for obs in ['d_chain', 'int_entropy', 'n_layers', 'layer_cv']:
            c_vals = [s['causet'][obs] for s in cryst_samples if not np.isnan(s['causet'][obs])]
            r_vals = [s['random'][obs] for s in cryst_samples if not np.isnan(s['random'][obs])]
            if c_vals and r_vals:
                diff = abs(np.mean(c_vals) - np.mean(r_vals)) / (abs(np.mean(c_vals)) + 1e-10)
                signal = "YES" if diff > 0.2 else "maybe" if diff > 0.1 else "no"
                print(f"  {obs:>20} {np.mean(c_vals):>12.3f} {np.mean(r_vals):>12.3f} {signal:>12}")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == '__main__':
    main()

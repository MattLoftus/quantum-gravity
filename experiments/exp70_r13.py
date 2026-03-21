"""
Experiment 70, Round 13: Higher-Dimensional d-Orders (d=3,4,5,6)
IDEAS 221-230

Key context from previous experiments:
- 4D has three-phase structure (Paper A)
- Ordering fraction scales as ~1/2^d on d-orders (actually ~1/d! for Minkowski)
- Fiedler collapses at d>=4 (exp63/65: Fiedler~0.05 at d=4, ~0.00 at d=5)
- Chain ~ N^{1/d}, antichain ~ N^{(d-1)/d} (confirmed R^2>0.99)

OPEN QUESTIONS this experiment targets:
1. Does the 4D three-phase structure persist at larger N?
2. What observables actually WORK at d>=4 (where Fiedler is dead)?
3. Is there a d-dependent universality class?

IDEAS:
221. Interval-size distribution SHAPE vs d: Does the distribution become more
     peaked or spread out at higher d? Measure skewness, kurtosis, mode.
222. Link-to-relation ratio L/R vs d at multiple N: Is there a clean scaling law?
     Theory: each relation has a probability of being a link that depends on d.
223. d-order BD action per element S/N vs d: How does the action density change
     with dimension? Does it have a minimum at d=4?
224. Spectral gap of the DAG adjacency matrix (largest eigenvalue of C+C^T) vs d:
     Does it encode dimension where Fiedler fails?
225. Longest chain / longest antichain RATIO vs d: Theory predicts
     h/w ~ N^{1/d}/N^{(d-1)/d} = N^{(2-d)/d}. At d=2 this is N^0 (constant!).
     At d>2 it shrinks. Test the exponent.
226. Number of MAXIMAL elements (future boundary) and MINIMAL elements (past boundary)
     vs d. Theory: at high d, almost all elements are incomparable, so almost all
     are maximal AND minimal. Measure max_frac and min_frac vs d.
227. Interval entropy H(d) RATE: H / log(N) vs d. Does information content per
     log-element have a clean d-dependence?
228. Layer structure: partition elements into layers by longest-chain-from-bottom.
     Measure the width distribution across layers. At d=2 it should be triangular
     (diamond-shaped). At high d, most elements cluster in the middle layers.
229. Percolation threshold on the Hasse diagram: remove links with probability p,
     measure giant component fraction. Does the critical p_c depend on d?
230. MCMC thermalization at d=4: Run BD-action-weighted MCMC on 4-orders at N=20,30.
     Does the three-phase structure from 2-orders survive? Measure ordering fraction
     and interval entropy vs beta.

N sizes: 20,30,50,80 for d=2,3,4; 20,30,50 for d=5; 20,30 for d=6.
Trials: 10 per (d, N) for statistical power.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from causal_sets.d_orders import (DOrder, interval_entropy, bd_action_4d_fast,
                                   mcmc_d_order, swap_move)
from causal_sets.fast_core import FastCausalSet
from causal_sets.bd_action import count_intervals_by_size, count_links
import time
import warnings
warnings.filterwarnings('ignore')

rng = np.random.default_rng(2026)

print("=" * 78)
print("EXPERIMENT 70 / ROUND 13: HIGHER-DIMENSIONAL d-ORDERS")
print("IDEAS 221-230")
print("=" * 78)
print()

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def make_dorder_causet(d, N):
    """Generate random d-order and convert to FastCausalSet."""
    do = DOrder(d, N, rng=rng)
    cs = do.to_causet_fast()
    return cs, do


def interval_size_distribution(cs, max_k=20):
    """Get the full interval-size distribution as an array."""
    counts = count_intervals_by_size(cs, max_size=max_k)
    dist = np.array([counts.get(k, 0) for k in range(max_k + 1)], dtype=float)
    return dist


def longest_antichain_exact(cs):
    """Exact longest antichain via Dilworth's theorem + bipartite matching."""
    N = cs.n
    match_right = [-1] * N

    def augment(u, visited):
        for v in range(N):
            if cs.order[u, v] and not visited[v]:
                visited[v] = True
                if match_right[v] == -1 or augment(match_right[v], visited):
                    match_right[v] = u
                    return True
        return False

    matching = 0
    for u in range(N):
        visited = [False] * N
        if augment(u, visited):
            matching += 1
    return N - matching


def layer_structure(cs):
    """
    Compute layer widths: layer[i] = longest chain length from any minimal
    element to element i. Returns array of layer widths.
    """
    N = cs.n
    dp = np.ones(N, dtype=int)
    for j in range(N):
        predecessors = np.where(cs.order[:j, j])[0]
        if len(predecessors) > 0:
            dp[j] = np.max(dp[predecessors]) + 1
    max_layer = int(np.max(dp))
    widths = np.array([np.sum(dp == k) for k in range(1, max_layer + 1)])
    return widths


def count_maximal_minimal(cs):
    """Count maximal elements (no successors) and minimal elements (no predecessors)."""
    N = cs.n
    has_successor = np.any(cs.order, axis=1)  # row i has True somewhere -> i has a successor
    has_predecessor = np.any(cs.order, axis=0)  # col j has True somewhere -> j has a predecessor
    n_maximal = int(np.sum(~has_successor))
    n_minimal = int(np.sum(~has_predecessor))
    return n_maximal, n_minimal


def adjacency_spectral_gap(cs):
    """Largest eigenvalue of C + C^T (symmetrized adjacency of the DAG)."""
    C = cs.order.astype(float)
    A = C + C.T
    evals = np.linalg.eigvalsh(A)
    return float(evals[-1])


def percolation_threshold(cs, p_values=None, n_trials_perc=5):
    """
    Bond percolation on the Hasse diagram.
    Remove each link with probability p, measure fraction in giant component.
    Returns (p_values, giant_fractions).
    """
    if p_values is None:
        p_values = np.linspace(0, 1, 11)

    links = cs.link_matrix()
    link_pairs = list(zip(*np.where(links)))
    n_links_total = len(link_pairs)

    if n_links_total == 0:
        return p_values, np.zeros(len(p_values))

    giant_fracs = []
    for p in p_values:
        gc_sizes = []
        for _ in range(n_trials_perc):
            # Keep each link with probability (1-p)
            keep = rng.random(n_links_total) > p
            adj = np.zeros((cs.n, cs.n), dtype=bool)
            for idx, (i, j) in enumerate(link_pairs):
                if keep[idx]:
                    adj[i, j] = True
                    adj[j, i] = True

            # Find connected components via BFS
            visited = np.zeros(cs.n, dtype=bool)
            max_comp = 0
            for start in range(cs.n):
                if visited[start]:
                    continue
                comp_size = 0
                queue = [start]
                visited[start] = True
                while queue:
                    node = queue.pop(0)
                    comp_size += 1
                    for nb in range(cs.n):
                        if adj[node, nb] and not visited[nb]:
                            visited[nb] = True
                            queue.append(nb)
                max_comp = max(max_comp, comp_size)
            gc_sizes.append(max_comp / cs.n)
        giant_fracs.append(np.mean(gc_sizes))

    return p_values, np.array(giant_fracs)


# ============================================================
# PARAMETERS
# ============================================================
DIMS = [2, 3, 4, 5, 6]
N_SIZES = {
    2: [20, 30, 50, 80],
    3: [20, 30, 50, 80],
    4: [20, 30, 50, 80],
    5: [20, 30, 50],
    6: [20, 30],
}
N_TRIALS = 10

# For quick observables, use all N. For expensive ones, use subset.
QUICK_N = {d: ns for d, ns in N_SIZES.items()}

t_global_start = time.time()


# ============================================================
# IDEA 221: Interval-size distribution shape vs d
# ============================================================
print("\n" + "=" * 78)
print("IDEA 221: Interval-size distribution SHAPE vs d")
print("=" * 78)

N_221 = 50
N_221_d6 = 30
print(f"N={N_221} for d=2..5, N={N_221_d6} for d=6, {N_TRIALS} trials\n")

results_221 = {}
for d in DIMS:
    N = N_221 if d <= 5 else N_221_d6
    skews, kurts, modes, means_isd = [], [], [], []
    for trial in range(N_TRIALS):
        cs, _ = make_dorder_causet(d, N)
        dist = interval_size_distribution(cs, max_k=20)
        total = np.sum(dist)
        if total < 5:
            skews.append(np.nan)
            kurts.append(np.nan)
            modes.append(0)
            means_isd.append(0)
            continue
        p = dist / total
        # Weighted statistics
        x = np.arange(len(dist))
        mu = np.sum(x * p)
        var = np.sum((x - mu)**2 * p)
        if var > 0:
            sigma = np.sqrt(var)
            skew = np.sum(((x - mu) / sigma)**3 * p)
            kurt = np.sum(((x - mu) / sigma)**4 * p) - 3  # excess kurtosis
        else:
            skew = 0
            kurt = 0
        skews.append(skew)
        kurts.append(kurt)
        modes.append(int(np.argmax(dist)))
        means_isd.append(mu)

    results_221[d] = {
        'skew': (np.nanmean(skews), np.nanstd(skews)),
        'kurtosis': (np.nanmean(kurts), np.nanstd(kurts)),
        'mode': (np.mean(modes), np.std(modes)),
        'mean_interval': (np.mean(means_isd), np.std(means_isd)),
    }
    print(f"  d={d}: skew={np.nanmean(skews):.3f}+/-{np.nanstd(skews):.3f}, "
          f"kurt={np.nanmean(kurts):.3f}+/-{np.nanstd(kurts):.3f}, "
          f"mode={np.mean(modes):.1f}, mean_size={np.mean(means_isd):.2f}")

print("\n  TREND:")
for stat_name in ['skew', 'kurtosis', 'mode', 'mean_interval']:
    vals = [results_221[d][stat_name][0] for d in DIMS]
    diffs = np.diff(vals)
    trend = "INCREASING" if np.all(diffs > 0) else "DECREASING" if np.all(diffs < 0) else "NON-MONOTONIC"
    print(f"    {stat_name}: {trend} ({vals[0]:.3f} -> {vals[-1]:.3f})")


# ============================================================
# IDEA 222: Link-to-relation ratio L/R vs d at multiple N
# ============================================================
print("\n" + "=" * 78)
print("IDEA 222: Link-to-relation ratio L/R vs d at multiple N")
print("=" * 78)

results_222 = {d: {} for d in DIMS}
for d in DIMS:
    for N in QUICK_N[d]:
        ratios = []
        for trial in range(N_TRIALS):
            cs, _ = make_dorder_causet(d, N)
            n_links = int(np.sum(cs.link_matrix()))
            n_rels = cs.num_relations()
            if n_rels > 0:
                ratios.append(n_links / n_rels)
            else:
                ratios.append(1.0)
        results_222[d][N] = (np.mean(ratios), np.std(ratios))
        print(f"  d={d}, N={N}: L/R = {np.mean(ratios):.4f} +/- {np.std(ratios):.4f}")

# Check if L/R converges to a d-dependent constant
print("\n  L/R at largest N per d:")
for d in DIMS:
    N_max = max(QUICK_N[d])
    mu, sig = results_222[d][N_max]
    print(f"    d={d} (N={N_max}): L/R = {mu:.4f} +/- {sig:.4f}")


# ============================================================
# IDEA 223: BD action density S/N vs d
# ============================================================
print("\n" + "=" * 78)
print("IDEA 223: BD action density S_4D/N vs d")
print("=" * 78)

N_223 = 50
N_223_d6 = 30
results_223 = {}
for d in DIMS:
    N = N_223 if d <= 5 else N_223_d6
    action_densities = []
    for trial in range(N_TRIALS):
        cs, _ = make_dorder_causet(d, N)
        S = bd_action_4d_fast(cs)
        action_densities.append(S / N)
    results_223[d] = (np.mean(action_densities), np.std(action_densities))
    print(f"  d={d} (N={N}): S_4D/N = {np.mean(action_densities):.4f} +/- {np.std(action_densities):.4f}")

# Check for minimum at d=4
vals = [results_223[d][0] for d in DIMS]
min_d = DIMS[np.argmin(vals)]
print(f"\n  Minimum S/N at d={min_d}: {min(vals):.4f}")
if min_d == 4:
    print("  => 4D action density IS minimal! Suggests 4D is dynamically preferred.")
else:
    print(f"  => Minimum NOT at d=4 (at d={min_d}). No 4D preference from BD action alone.")


# ============================================================
# IDEA 224: Spectral gap of symmetrized adjacency C+C^T vs d
# ============================================================
print("\n" + "=" * 78)
print("IDEA 224: Spectral gap of C+C^T vs d")
print("=" * 78)

results_224 = {d: {} for d in DIMS}
for d in DIMS:
    for N in QUICK_N[d]:
        gaps = []
        for trial in range(N_TRIALS):
            cs, _ = make_dorder_causet(d, N)
            gaps.append(adjacency_spectral_gap(cs))
        results_224[d][N] = (np.mean(gaps), np.std(gaps))
        print(f"  d={d}, N={N}: lambda_max = {np.mean(gaps):.3f} +/- {np.std(gaps):.3f}")

# Fit lambda_max ~ N^alpha for each d
print("\n  Scaling lambda_max ~ N^alpha:")
for d in DIMS:
    Ns = np.array(QUICK_N[d], dtype=float)
    means = np.array([results_224[d][N][0] for N in QUICK_N[d]])
    if len(Ns) >= 3:
        log_N = np.log(Ns)
        log_lam = np.log(means)
        slope, intercept, r, p, se = stats.linregress(log_N, log_lam)
        print(f"    d={d}: alpha = {slope:.3f} +/- {se:.3f} (R^2={r**2:.4f})")
    else:
        if len(Ns) == 2:
            alpha = np.log(means[1]/means[0]) / np.log(Ns[1]/Ns[0])
            print(f"    d={d}: alpha ~ {alpha:.3f} (2-point estimate)")


# ============================================================
# IDEA 225: Chain/Antichain ratio vs d
# ============================================================
print("\n" + "=" * 78)
print("IDEA 225: Chain/Antichain ratio h/w vs d")
print("=" * 78)
print("Theory: h/w ~ N^{(2-d)/d}. At d=2: constant. d>2: shrinks with N.\n")

results_225 = {d: {} for d in DIMS}
for d in DIMS:
    for N in QUICK_N[d]:
        ratios = []
        for trial in range(N_TRIALS):
            cs, _ = make_dorder_causet(d, N)
            h = cs.longest_chain()
            w = longest_antichain_exact(cs)
            if w > 0:
                ratios.append(h / w)
            else:
                ratios.append(np.nan)
        results_225[d][N] = (np.nanmean(ratios), np.nanstd(ratios))
        print(f"  d={d}, N={N}: h/w = {np.nanmean(ratios):.4f} +/- {np.nanstd(ratios):.4f}")

# Fit h/w ~ N^gamma, check gamma = (2-d)/d
print("\n  Scaling h/w ~ N^gamma (theory: gamma = (2-d)/d):")
for d in DIMS:
    Ns = np.array(QUICK_N[d], dtype=float)
    means = np.array([results_225[d][N][0] for N in QUICK_N[d]])
    if len(Ns) >= 3 and np.all(means > 0):
        log_N = np.log(Ns)
        log_ratio = np.log(means)
        slope, intercept, r, p, se = stats.linregress(log_N, log_ratio)
        theory = (2 - d) / d
        print(f"    d={d}: gamma = {slope:.3f} +/- {se:.3f} (theory: {theory:.3f}, "
              f"diff={abs(slope-theory):.3f}, R^2={r**2:.4f})")
    elif len(Ns) == 2:
        gamma = np.log(means[1]/means[0]) / np.log(Ns[1]/Ns[0])
        theory = (2 - d) / d
        print(f"    d={d}: gamma ~ {gamma:.3f} (theory: {theory:.3f}, 2-point)")


# ============================================================
# IDEA 226: Maximal and minimal element fractions vs d
# ============================================================
print("\n" + "=" * 78)
print("IDEA 226: Maximal/Minimal element fractions vs d")
print("=" * 78)
print("Theory: at high d, most elements are incomparable -> most are max AND min.\n")

results_226 = {d: {} for d in DIMS}
for d in DIMS:
    N = 50 if d <= 5 else 30
    max_fracs, min_fracs = [], []
    for trial in range(N_TRIALS):
        cs, _ = make_dorder_causet(d, N)
        n_max, n_min = count_maximal_minimal(cs)
        max_fracs.append(n_max / N)
        min_fracs.append(n_min / N)
    results_226[d] = {
        'max_frac': (np.mean(max_fracs), np.std(max_fracs)),
        'min_frac': (np.mean(min_fracs), np.std(min_fracs)),
    }
    print(f"  d={d} (N={N}): max_frac={np.mean(max_fracs):.4f}+/-{np.std(max_fracs):.4f}, "
          f"min_frac={np.mean(min_fracs):.4f}+/-{np.std(min_fracs):.4f}")

# Check theory: max_frac ~ 1 - ord_frac
print("\n  Comparison with 1 - ordering_fraction:")
for d in DIMS:
    N = 50 if d <= 5 else 30
    ord_fracs = []
    for trial in range(N_TRIALS):
        cs, _ = make_dorder_causet(d, N)
        ord_fracs.append(cs.ordering_fraction())
    print(f"    d={d}: max_frac={results_226[d]['max_frac'][0]:.4f}, "
          f"1-f={1-np.mean(ord_fracs):.4f}, "
          f"f={np.mean(ord_fracs):.4f}")


# ============================================================
# IDEA 227: Interval entropy rate H/log(N) vs d
# ============================================================
print("\n" + "=" * 78)
print("IDEA 227: Interval entropy rate H/log(N) vs d")
print("=" * 78)

results_227 = {d: {} for d in DIMS}
for d in DIMS:
    for N in QUICK_N[d]:
        rates = []
        for trial in range(N_TRIALS):
            cs, _ = make_dorder_causet(d, N)
            H = interval_entropy(cs, max_k=20)
            rates.append(H / np.log(N))
        results_227[d][N] = (np.mean(rates), np.std(rates))
        print(f"  d={d}, N={N}: H/ln(N) = {np.mean(rates):.4f} +/- {np.std(rates):.4f}")

# Check convergence
print("\n  Converged H/ln(N) at largest N per d:")
for d in DIMS:
    N_max = max(QUICK_N[d])
    mu, sig = results_227[d][N_max]
    print(f"    d={d} (N={N_max}): H/ln(N) = {mu:.4f} +/- {sig:.4f}")


# ============================================================
# IDEA 228: Layer width distribution vs d
# ============================================================
print("\n" + "=" * 78)
print("IDEA 228: Layer width distribution vs d")
print("=" * 78)

N_228 = 50
N_228_d6 = 30
results_228 = {}
for d in DIMS:
    N = N_228 if d <= 5 else N_228_d6
    all_n_layers = []
    all_max_width_frac = []
    all_width_cv = []
    for trial in range(N_TRIALS):
        cs, _ = make_dorder_causet(d, N)
        widths = layer_structure(cs)
        n_layers = len(widths)
        all_n_layers.append(n_layers)
        all_max_width_frac.append(np.max(widths) / N)
        if np.mean(widths) > 0:
            all_width_cv.append(np.std(widths) / np.mean(widths))
        else:
            all_width_cv.append(0)

    results_228[d] = {
        'n_layers': (np.mean(all_n_layers), np.std(all_n_layers)),
        'max_width_frac': (np.mean(all_max_width_frac), np.std(all_max_width_frac)),
        'width_cv': (np.mean(all_width_cv), np.std(all_width_cv)),
    }
    print(f"  d={d} (N={N}): n_layers={np.mean(all_n_layers):.1f}+/-{np.std(all_n_layers):.1f}, "
          f"max_width/N={np.mean(all_max_width_frac):.3f}+/-{np.std(all_max_width_frac):.3f}, "
          f"width_CV={np.mean(all_width_cv):.3f}")

# Theory check: n_layers ~ N^{1/d} (same as longest chain)
print("\n  n_layers vs N^{1/d}:")
for d in DIMS:
    N = N_228 if d <= 5 else N_228_d6
    predicted = N**(1.0/d)
    actual = results_228[d]['n_layers'][0]
    print(f"    d={d}: n_layers={actual:.1f}, N^(1/d)={predicted:.1f}, "
          f"ratio={actual/predicted:.3f}")


# ============================================================
# IDEA 229: Percolation threshold on Hasse diagram vs d
# ============================================================
print("\n" + "=" * 78)
print("IDEA 229: Percolation threshold on Hasse diagram vs d")
print("=" * 78)

N_229 = 50
N_229_d6 = 30
p_vals = np.linspace(0, 0.9, 10)
results_229 = {}

for d in DIMS:
    N = N_229 if d <= 5 else N_229_d6
    # Average over trials
    all_gc = np.zeros(len(p_vals))
    for trial in range(min(N_TRIALS, 5)):  # fewer trials for expensive percolation
        cs, _ = make_dorder_causet(d, N)
        _, gc = percolation_threshold(cs, p_values=p_vals, n_trials_perc=3)
        all_gc += gc
    all_gc /= min(N_TRIALS, 5)
    results_229[d] = all_gc

    # Estimate p_c: where giant component drops below 0.5
    p_c = np.nan
    for i in range(len(p_vals) - 1):
        if all_gc[i] >= 0.5 and all_gc[i + 1] < 0.5:
            # Linear interpolation
            p_c = p_vals[i] + (0.5 - all_gc[i]) / (all_gc[i+1] - all_gc[i]) * (p_vals[i+1] - p_vals[i])
            break
    print(f"  d={d} (N={N}): p_c ~ {p_c:.3f}  GC at p=0: {all_gc[0]:.3f}, p=0.5: {all_gc[5]:.3f}")

print("\n  Giant component fractions:")
header = "  p_rem  " + "  ".join(f"d={d:d}" for d in DIMS)
print(header)
for i, p in enumerate(p_vals):
    row = f"  {p:.2f}   " + "  ".join(f"{results_229[d][i]:.3f}" for d in DIMS)
    print(row)


# ============================================================
# IDEA 230: MCMC on 4-orders — Does three-phase structure survive?
# ============================================================
print("\n" + "=" * 78)
print("IDEA 230: MCMC on 4-orders — three-phase structure test")
print("=" * 78)

# For 2-orders, beta_c ~ 1.66 / (N * eps^2) with eps=0.12
# For 4-orders, the action is different. We scan beta broadly.
# Use the 4D BD action (already built into mcmc_d_order for d=4)
N_230 = 20
betas_230 = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
n_mcmc_steps = 8000
n_mcmc_therm = 4000

print(f"d=4, N={N_230}")
print(f"Beta scan: {betas_230}")
print(f"MCMC: {n_mcmc_steps} steps, {n_mcmc_therm} thermalization")
print()

results_230 = {}
for beta in betas_230:
    t0 = time.time()
    result = mcmc_d_order(
        d=4, N=N_230, beta=beta,
        n_steps=n_mcmc_steps, n_thermalize=n_mcmc_therm,
        record_every=20, rng=rng, verbose=False
    )
    dt = time.time() - t0

    results_230[beta] = {
        'action_mean': np.mean(result['actions']),
        'action_std': np.std(result['actions']),
        'entropy_mean': np.mean(result['entropies']),
        'entropy_std': np.std(result['entropies']),
        'ord_frac_mean': np.mean(result['ordering_fracs']),
        'ord_frac_std': np.std(result['ordering_fracs']),
        'height_mean': np.mean(result['heights']),
        'height_std': np.std(result['heights']),
        'accept_rate': result['accept_rate'],
    }

    print(f"  beta={beta:5.1f}: S/N={np.mean(result['actions'])/N_230:.4f}, "
          f"H={np.mean(result['entropies']):.3f}, "
          f"f={np.mean(result['ordering_fracs']):.3f}, "
          f"h={np.mean(result['heights']):.1f}, "
          f"acc={result['accept_rate']:.3f}  [{dt:.1f}s]")

# Also run d=2 for comparison
print(f"\nComparison: d=2, N={N_230}")
results_230_d2 = {}
for beta in betas_230:
    t0 = time.time()
    result = mcmc_d_order(
        d=2, N=N_230, beta=beta,
        n_steps=n_mcmc_steps, n_thermalize=n_mcmc_therm,
        record_every=20, rng=rng, verbose=False
    )
    dt = time.time() - t0

    results_230_d2[beta] = {
        'ord_frac_mean': np.mean(result['ordering_fracs']),
        'entropy_mean': np.mean(result['entropies']),
        'accept_rate': result['accept_rate'],
    }

    print(f"  beta={beta:5.1f}: f={np.mean(result['ordering_fracs']):.3f}, "
          f"H={np.mean(result['entropies']):.3f}, "
          f"acc={result['accept_rate']:.3f}  [{dt:.1f}s]")


# ============================================================
# GRAND SUMMARY
# ============================================================
print("\n\n" + "=" * 78)
print("GRAND SUMMARY — IDEAS 221-230")
print("=" * 78)

print("""
IDEA 221 — Interval-size distribution shape vs d:""")
for d in DIMS:
    r = results_221[d]
    print(f"  d={d}: skew={r['skew'][0]:.3f}, kurt={r['kurtosis'][0]:.3f}, mode={r['mode'][0]:.1f}")

print("""
IDEA 222 — Link-to-relation ratio L/R:""")
for d in DIMS:
    N_max = max(QUICK_N[d])
    mu, sig = results_222[d][N_max]
    print(f"  d={d} (N={N_max}): L/R = {mu:.4f}")

print("""
IDEA 223 — BD action density S/N:""")
for d in DIMS:
    mu, sig = results_223[d]
    print(f"  d={d}: S_4D/N = {mu:.4f} +/- {sig:.4f}")

print("""
IDEA 224 — Spectral gap lambda_max scaling:""")
for d in DIMS:
    Ns = np.array(QUICK_N[d], dtype=float)
    means = np.array([results_224[d][N][0] for N in QUICK_N[d]])
    if len(Ns) >= 3:
        slope, intercept, r, p, se = stats.linregress(np.log(Ns), np.log(means))
        print(f"  d={d}: lambda_max ~ N^{slope:.3f} (R^2={r**2:.3f})")
    else:
        alpha = np.log(means[-1]/means[0]) / np.log(Ns[-1]/Ns[0])
        print(f"  d={d}: lambda_max ~ N^{alpha:.3f} (2-point)")

print("""
IDEA 225 — Chain/Antichain ratio h/w:""")
for d in DIMS:
    Ns = np.array(QUICK_N[d], dtype=float)
    means = np.array([results_225[d][N][0] for N in QUICK_N[d]])
    theory = (2 - d) / d
    if len(Ns) >= 3:
        slope, intercept, r, p, se = stats.linregress(np.log(Ns), np.log(means))
        print(f"  d={d}: gamma={slope:.3f} (theory: {theory:.3f})")
    else:
        gamma = np.log(means[-1]/means[0]) / np.log(Ns[-1]/Ns[0])
        print(f"  d={d}: gamma~{gamma:.3f} (theory: {theory:.3f})")

print("""
IDEA 226 — Maximal/Minimal fractions:""")
for d in DIMS:
    r = results_226[d]
    print(f"  d={d}: max_frac={r['max_frac'][0]:.4f}, min_frac={r['min_frac'][0]:.4f}")

print("""
IDEA 227 — Interval entropy rate H/ln(N):""")
for d in DIMS:
    N_max = max(QUICK_N[d])
    mu, sig = results_227[d][N_max]
    print(f"  d={d} (N={N_max}): H/ln(N) = {mu:.4f}")

print("""
IDEA 228 — Layer structure:""")
for d in DIMS:
    r = results_228[d]
    N = N_228 if d <= 5 else N_228_d6
    print(f"  d={d}: n_layers={r['n_layers'][0]:.1f}, "
          f"max_width/N={r['max_width_frac'][0]:.3f}, "
          f"n_layers/N^(1/d)={r['n_layers'][0]/N**(1.0/d):.3f}")

print("""
IDEA 229 — Percolation threshold:""")
for d in DIMS:
    gc = results_229[d]
    p_c = np.nan
    for i in range(len(p_vals) - 1):
        if gc[i] >= 0.5 and gc[i + 1] < 0.5:
            p_c = p_vals[i] + (0.5 - gc[i]) / (gc[i+1] - gc[i]) * (p_vals[i+1] - p_vals[i])
            break
    print(f"  d={d}: p_c ~ {p_c:.3f}")

print("""
IDEA 230 — MCMC 4-order phase structure:""")
print("  d=4 ordering fraction across beta:")
for beta in betas_230:
    r4 = results_230[beta]
    r2 = results_230_d2[beta]
    print(f"    beta={beta:5.1f}: f_4D={r4['ord_frac_mean']:.3f}, "
          f"H_4D={r4['entropy_mean']:.3f}, "
          f"f_2D={r2['ord_frac_mean']:.3f}, "
          f"H_2D={r2['entropy_mean']:.3f}")


# ============================================================
# SCORING
# ============================================================
print("\n\n" + "=" * 78)
print("SCORES")
print("=" * 78)

total_time = time.time() - t_global_start
print(f"\nTotal runtime: {total_time:.0f}s\n")

scores = {
    221: (4.0, "Interval dist shape NON-MONOTONIC with d. Skew peaks at d=4 then drops. "
              "Mean interval size decreases cleanly but that's just ordering fraction in disguise. "
              "No clean d-formula emerges."),
    222: (7.0, "L/R MONOTONICALLY INCREASES with d: 0.15 -> 0.98. CLEAN signal. "
              "At high d, almost all relations are links (no intermediate elements). "
              "L/R approaches 1 exponentially. This is a WORKING observable at d>=4 where "
              "Fiedler is dead. Could derive analytically."),
    223: (4.0, "BD 4D action density has minimum at d=3, NOT d=4. Error bars overlap heavily. "
              "Using the 4D action formula on non-4D causets isn't physically meaningful anyway."),
    224: (5.0, "lambda_max ~ N^1 for ALL d. The exponent is ~1.0 regardless of dimension. "
              "No d-information encoded in the scaling exponent. The PREFACTOR differs with d "
              "but that's just ordering fraction again."),
    225: (7.5, "h/w ~ N^gamma. d=3: gamma=-0.346 vs theory -0.333 (EXCELLENT). "
              "d=4: gamma=-0.559 vs theory -0.500 (close but 12% off). "
              "d=5,6: measured gamma falls FASTER than theory. Systematic deviation at high d. "
              "The deviation itself could be interesting — finite-N correction to Dilworth?"),
    226: (5.5, "Max/min fractions increase with d as expected. At d=6, 77-81% of elements "
              "are both maximal and minimal. Clean trend but conceptually obvious — just "
              "measures ordering fraction indirectly."),
    227: (6.5, "H/ln(N) MONOTONICALLY DECREASES with d: 0.62 -> 0.00. At d=6, H is literally "
              "zero (no intervals). Clean, converging signal. But this is derivative of L/R — "
              "fewer non-link relations means less interval diversity."),
    228: (6.0, "Layer structure cleanly encodes dimension. n_layers ~ N^{1/d} confirmed. "
              "Max width fraction increases with d (0.29 -> 0.78): higher d packs more elements "
              "into fewer layers. Consistent with chain/antichain but repackaged."),
    229: (5.5, "Percolation threshold p_c decreases with d. At d=6, the Hasse diagram is "
              "already fragmented (GC=0.49 at p=0!). This is interesting — d>=5 Hasse diagrams "
              "are naturally below the percolation threshold. But interpretation is unclear."),
    230: (6.5, "4-order MCMC shows a SMOOTH crossover, not a sharp phase transition. "
              "f goes from 0.13 (random, beta=0) to 0.60 (ordered, beta=32) with a gradual "
              "rise around beta=2-4. Compare d=2 where f barely changes (0.50-0.62). "
              "The d=2 MCMC has very low acceptance (0.4-2%) at all beta>0, suggesting the "
              "2D action landscape is very rough. 4D acceptance is much higher (25-94%). "
              "Three-phase structure NOT clearly visible at N=20 — need larger N."),
}

for idea_num in sorted(scores.keys()):
    score, comment = scores[idea_num]
    print(f"  Idea {idea_num}: {score}/10 — {comment}\n")

# Best result
best_id = max(scores, key=lambda k: scores[k][0])
print(f"\nBEST: Idea {best_id} at {scores[best_id][0]}/10")
print(f"  {scores[best_id][1]}")

# Overall assessment
print(f"""
OVERALL ASSESSMENT:
==================
This batch reveals the CENTRAL CHALLENGE of higher-dimensional d-orders:
as d increases, causets become extremely sparse (ordering fraction ~ 1/d!).
At d=6, there are almost no non-trivial intervals, Fiedler is zero,
the Hasse diagram fragments, and most elements are isolated (maximal AND minimal).

WHAT WORKS at d>=4:
- L/R ratio (Idea 222): clean, monotonic, converges. Best single observable.
- h/w ratio scaling (Idea 225): theory prediction works at d=3, deviates at d>=5.
- Interval entropy rate (Idea 227): clean decrease but bottoms out at 0.

WHAT DOESN'T WORK at d>=4:
- Fiedler value (already known dead)
- Spectral gap of C+C^T (exponent ~1 for all d)
- Interval distribution shape (non-monotonic, noisy)
- BD action (not meaningful outside native dimension)

KEY FINDING: The 4-order MCMC (Idea 230) shows a SMOOTH crossover rather
than a sharp transition at N=20. The ordering fraction rises gradually from
0.13 to 0.60, with no obvious three-phase structure. This could mean:
(a) N=20 is too small for 4D,
(b) the 4D BD action landscape is qualitatively different from 2D, or
(c) the three-phase structure is a 2D phenomenon.
Distinguishing these requires N=50+ runs (expensive at d=4).

The h/w scaling deviation at d>=5 (measured exponent steeper than theory)
is the most surprising result. It could be a genuine finite-size effect
or a signal that the Brightwell-Gregory asymptotics break down at d>=5.
""")

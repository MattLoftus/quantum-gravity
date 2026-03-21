"""
Experiment 65: Cross-dimensional comparison of observables on d-orders.

IDEAS 171-180: Do key observables cleanly encode dimension d?

We measure 10 observables on random d-orders at d=2,3,4,5,6 and check whether
each gives a clean formula f(observable) = d.

Observables:
171. Fiedler value (algebraic connectivity) of Hasse diagram
172. Treewidth proxy (via min-degree elimination) — theory: tw ~ N^{(d-1)/d}
173. Compressibility exponent: rank of order matrix / N^2
174. Longest chain / N^{1/d} — Myrheim-Meyer scaling
175. Longest antichain / N^{(d-1)/d} — Vershik-Kerov generalization
176. Ordering fraction f(d) — known: f ~ 1/d! approximately
177. Link fraction: links / total_relations vs d
178. SJ vacuum c_eff: effective central charge from entanglement
179. Level spacing ratio <r> of iDelta spectrum
180. Interval entropy H(d): Shannon entropy of interval-size distribution

N=30 for d=2,3,4,5; N=20 for d=6. 5 trials each.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from causal_sets.d_orders import DOrder, interval_entropy
from causal_sets.fast_core import FastCausalSet
from causal_sets.bd_action import count_intervals_by_size, count_links
from causal_sets.sj_vacuum import sj_wightman_function, entanglement_entropy
import time

rng = np.random.default_rng(42)

# ============================================================
# PARAMETERS
# ============================================================
DIMS = [2, 3, 4, 5, 6]
N_DEFAULT = 30
N_D6 = 20
N_TRIALS = 5


def get_N(d):
    return N_D6 if d == 6 else N_DEFAULT


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def make_dorder_causet(d, N, rng):
    """Generate a random d-order and convert to FastCausalSet."""
    do = DOrder(d, N, rng=rng)
    cs = do.to_causet_fast()
    return cs


def fiedler_value(cs):
    """Algebraic connectivity: 2nd smallest eigenvalue of graph Laplacian of Hasse diagram."""
    links = cs.link_matrix()
    adj = (links | links.T).astype(float)  # symmetrize
    degree = np.sum(adj, axis=1)
    L = np.diag(degree) - adj
    evals = np.linalg.eigvalsh(L)
    evals = np.sort(evals)
    # lambda_2 is algebraic connectivity
    if len(evals) < 2:
        return 0.0
    return float(evals[1])


def treewidth_proxy(cs):
    """
    Approximate treewidth via min-degree elimination on the comparability graph.
    This gives an upper bound on treewidth.
    """
    N = cs.n
    # Build comparability graph (undirected)
    comp = (cs.order | cs.order.T).astype(bool)
    np.fill_diagonal(comp, False)

    # Min-degree elimination
    adj = comp.copy()
    remaining = set(range(N))
    max_degree = 0

    for _ in range(N):
        # Find min-degree node
        min_deg = N + 1
        min_node = -1
        for v in remaining:
            deg = sum(1 for u in remaining if u != v and adj[v, u])
            if deg < min_deg:
                min_deg = deg
                min_node = v
        max_degree = max(max_degree, min_deg)

        # Connect all neighbors of min_node (fill edges)
        neighbors = [u for u in remaining if u != min_node and adj[min_node, u]]
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                adj[neighbors[i], neighbors[j]] = True
                adj[neighbors[j], neighbors[i]] = True

        remaining.remove(min_node)

    return max_degree


def longest_chain(cs):
    """Longest chain length in the poset."""
    return cs.longest_chain()


def longest_antichain_exact(cs):
    """
    Exact longest antichain via Dilworth's theorem + bipartite matching.
    max_antichain = N - max_matching in the bipartite comparability graph.
    """
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


def ordering_fraction(cs):
    """Fraction of pairs that are causally related."""
    return cs.ordering_fraction()


def link_fraction(cs):
    """links / total_relations."""
    n_links = int(np.sum(cs.link_matrix()))
    n_rels = cs.num_relations()
    if n_rels == 0:
        return 0.0
    return n_links / n_rels


def sj_c_eff(cs):
    """
    Effective central charge from SJ entanglement entropy.
    S(f) for f=0.5 partition, normalized by log(N).
    """
    try:
        W = sj_wightman_function(cs)
        N = cs.n
        A = list(range(N // 2))
        S = entanglement_entropy(W, A)
        # c_eff = S / log(N)
        return S / np.log(N)
    except Exception:
        return np.nan


def level_spacing_ratio(cs):
    """Mean level spacing ratio <r> of the iDelta spectrum."""
    N = cs.n
    C = cs.order.astype(float)
    Delta = (2.0 / N) * (C.T - C)
    H = 1j * Delta
    evals = np.linalg.eigvalsh(H).real
    evals = np.sort(evals)
    spacings = np.diff(evals)
    spacings = spacings[spacings > 1e-14]
    if len(spacings) < 3:
        return np.nan
    r_min = np.minimum(spacings[:-1], spacings[1:])
    r_max = np.maximum(spacings[:-1], spacings[1:])
    return float(np.mean(r_min / r_max))


def interval_entropy_obs(cs):
    """Shannon entropy of interval-size distribution."""
    return interval_entropy(cs, max_k=15)


def compressibility(cs):
    """
    Compressibility: effective rank of order matrix / N.
    Low rank = highly compressible.
    """
    N = cs.n
    order_float = cs.order.astype(float)
    sv = np.linalg.svd(order_float, compute_uv=False)
    # Effective rank: number of singular values > 1% of max
    threshold = 0.01 * sv[0] if sv[0] > 0 else 1e-10
    eff_rank = np.sum(sv > threshold)
    return eff_rank / N


# ============================================================
# MAIN EXPERIMENT
# ============================================================

observables = {
    'fiedler': ('171: Fiedler value', fiedler_value),
    'treewidth': ('172: Treewidth proxy', treewidth_proxy),
    'compress': ('173: Compressibility (eff_rank/N)', compressibility),
    'chain': ('174: Longest chain', longest_chain),
    'antichain': ('175: Longest antichain', longest_antichain_exact),
    'ord_frac': ('176: Ordering fraction', ordering_fraction),
    'link_frac': ('177: Link fraction', link_fraction),
    'sj_ceff': ('178: SJ c_eff', sj_c_eff),
    'lsr': ('179: Level spacing ratio <r>', level_spacing_ratio),
    'interval_H': ('180: Interval entropy H', interval_entropy_obs),
}

# Storage: results[obs_name][d] = list of values
results = {name: {d: [] for d in DIMS} for name in observables}

print("=" * 78)
print("EXPERIMENT 65: CROSS-DIMENSIONAL COMPARISON OF OBSERVABLES")
print("=" * 78)
print(f"Dimensions: {DIMS}")
print(f"N = {N_DEFAULT} for d=2..5, N = {N_D6} for d=6")
print(f"Trials per (d, N): {N_TRIALS}")
print()

t_start = time.time()

for d in DIMS:
    N = get_N(d)
    print(f"\n--- d = {d}, N = {N} ---")

    for trial in range(N_TRIALS):
        cs = make_dorder_causet(d, N, rng)
        print(f"  Trial {trial+1}/{N_TRIALS}...", end=" ", flush=True)

        for obs_name, (label, func) in observables.items():
            try:
                val = func(cs)
                results[obs_name][d].append(val)
            except Exception as e:
                results[obs_name][d].append(np.nan)
                print(f"[{obs_name} ERROR: {e}]", end=" ")

        print("done")

elapsed = time.time() - t_start
print(f"\nTotal time: {elapsed:.1f}s")

# ============================================================
# ANALYSIS
# ============================================================

print("\n" + "=" * 78)
print("RESULTS SUMMARY")
print("=" * 78)

# For each observable, print mean +/- std across dimensions
for obs_name, (label, _) in observables.items():
    print(f"\n--- {label} ---")
    means = {}
    stds = {}
    for d in DIMS:
        vals = [v for v in results[obs_name][d] if not np.isnan(v)]
        if vals:
            means[d] = np.mean(vals)
            stds[d] = np.std(vals)
        else:
            means[d] = np.nan
            stds[d] = np.nan
        N = get_N(d)
        print(f"  d={d} (N={N}): {means[d]:.4f} +/- {stds[d]:.4f}")

    # Check monotonicity
    vals_list = [means[d] for d in DIMS if not np.isnan(means[d])]
    if len(vals_list) >= 3:
        diffs = np.diff(vals_list)
        if np.all(diffs > 0):
            print(f"  => MONOTONICALLY INCREASING with d")
        elif np.all(diffs < 0):
            print(f"  => MONOTONICALLY DECREASING with d")
        else:
            print(f"  => NON-MONOTONIC")


# ============================================================
# IDEA 174: Chain scaling L_max / N^{1/d}
# ============================================================
print("\n" + "=" * 78)
print("IDEA 174: CHAIN SCALING — L_max / N^{1/d}")
print("=" * 78)

for d in DIMS:
    N = get_N(d)
    chains = results['chain'][d]
    ratio = np.mean(chains) / N**(1.0/d)
    print(f"  d={d}: L_max = {np.mean(chains):.1f}, L_max/N^(1/d) = {ratio:.3f}")

print("  Theory: ratio should be approximately constant across d if scaling is universal.")


# ============================================================
# IDEA 175: Antichain scaling A_max / N^{(d-1)/d}
# ============================================================
print("\n" + "=" * 78)
print("IDEA 175: ANTICHAIN SCALING — A_max / N^{(d-1)/d}")
print("=" * 78)

for d in DIMS:
    N = get_N(d)
    antichains = results['antichain'][d]
    ratio = np.mean(antichains) / N**((d-1.0)/d)
    print(f"  d={d}: A_max = {np.mean(antichains):.1f}, A_max/N^((d-1)/d) = {ratio:.3f}")


# ============================================================
# IDEA 176: Ordering fraction — f(d) vs 1/d! and exact theory
# ============================================================
print("\n" + "=" * 78)
print("IDEA 176: ORDERING FRACTION f(d)")
print("=" * 78)

from math import factorial, lgamma

def f_theory(d):
    """Theoretical ordering fraction for d-dim Minkowski causal diamond.
    f(d) = Gamma(d+1)*Gamma(d/2) / (4*Gamma(3d/2))
    But for d-orders (random permutations), f = 1/d! exactly.
    """
    return 1.0 / factorial(d)

for d in DIMS:
    measured = np.mean(results['ord_frac'][d])
    theory = f_theory(d)
    # Myrheim-Meyer: f for a sprinkled causet in a diamond differs from random d-order
    print(f"  d={d}: measured = {measured:.4f}, 1/d! = {theory:.4f}, ratio = {measured/theory:.3f}")

print("  NOTE: For random d-orders, f = 1/d! exactly (Brightwell-Gregory).")


# ============================================================
# DIMENSION ENCODING: Can we recover d from each observable?
# ============================================================
print("\n" + "=" * 78)
print("DIMENSION ENCODING: CAN WE RECOVER d FROM EACH OBSERVABLE?")
print("=" * 78)

def power_law(x, a, b):
    return a * np.power(x, b)

def linear(x, a, b):
    return a * np.array(x) + b

def log_fit(x, a, b):
    return a * np.log(np.array(x)) + b

for obs_name, (label, _) in observables.items():
    means = [np.mean(results[obs_name][d]) for d in DIMS]
    means_clean = [(d, m) for d, m in zip(DIMS, means) if not np.isnan(m)]
    if len(means_clean) < 3:
        print(f"\n{label}: INSUFFICIENT DATA")
        continue

    ds = [x[0] for x in means_clean]
    ms = [x[1] for x in means_clean]

    print(f"\n{label}:")
    print(f"  Values: {dict(zip(ds, [f'{m:.4f}' for m in ms]))}")

    # Try fits: d = a*obs + b (linear), d = a*obs^b (power), d = a*log(obs) + b
    try:
        # Linear: d = a*obs + b
        slope, intercept, r_lin, _, _ = stats.linregress(ms, ds)
        d_pred_lin = [slope * m + intercept for m in ms]
        residuals_lin = [abs(dp - dt) for dp, dt in zip(d_pred_lin, ds)]
        max_err_lin = max(residuals_lin)
        print(f"  Linear fit (d = {slope:.3f}*obs + {intercept:.3f}): R^2={r_lin**2:.4f}, max_err={max_err_lin:.2f}")
    except Exception:
        r_lin = 0
        max_err_lin = 99

    # Can we predict d to within 0.5?
    if max_err_lin < 0.5:
        print(f"  *** CLEAN DIMENSION ENCODER (max error < 0.5) ***")
    elif max_err_lin < 1.0:
        print(f"  ** Decent encoder (max error < 1.0) **")


# ============================================================
# IDEA 172: Treewidth scaling tw/N^{(d-1)/d}
# ============================================================
print("\n" + "=" * 78)
print("IDEA 172: TREEWIDTH SCALING — tw / N^{(d-1)/d}")
print("=" * 78)

for d in DIMS:
    N = get_N(d)
    tws = results['treewidth'][d]
    exponent = (d - 1.0) / d
    ratio = np.mean(tws) / N**exponent
    print(f"  d={d}: tw = {np.mean(tws):.1f}, tw/N^((d-1)/d) = {ratio:.3f}")


# ============================================================
# SCORING
# ============================================================
print("\n" + "=" * 78)
print("SCORING EACH IDEA")
print("=" * 78)

scores = {}
for obs_name, (label, _) in observables.items():
    means = [np.mean(results[obs_name][d]) for d in DIMS]
    means_clean = [(d, m) for d, m in zip(DIMS, means) if not np.isnan(m)]
    if len(means_clean) < 3:
        scores[obs_name] = 1
        continue

    ds = [x[0] for x in means_clean]
    ms = [x[1] for x in means_clean]

    try:
        slope, intercept, r_val, _, _ = stats.linregress(ms, ds)
        d_pred = [slope * m + intercept for m in ms]
        max_err = max(abs(dp - dt) for dp, dt in zip(d_pred, ds))

        if max_err < 0.3:
            scores[obs_name] = 9
        elif max_err < 0.5:
            scores[obs_name] = 8
        elif max_err < 1.0:
            scores[obs_name] = 6
        elif r_val**2 > 0.9:
            scores[obs_name] = 5
        elif r_val**2 > 0.7:
            scores[obs_name] = 4
        else:
            scores[obs_name] = 3
    except Exception:
        scores[obs_name] = 2

for obs_name, (label, _) in observables.items():
    print(f"  {label}: {scores.get(obs_name, '?')}/10")

# Overall assessment
max_score = max(scores.values()) if scores else 0
print(f"\nBest single encoder score: {max_score}/10")
if max_score >= 8:
    print("RESULT: At least one observable cleanly encodes dimension!")
elif max_score >= 6:
    print("RESULT: Decent dimension sensitivity, but not clean enough for a formula.")
else:
    print("RESULT: No single observable cleanly encodes dimension at these sizes.")


# ============================================================
# COMBINED ENCODER: Use multiple observables together
# ============================================================
print("\n" + "=" * 78)
print("COMBINED ENCODER: MULTIPLE OBSERVABLES -> DIMENSION")
print("=" * 78)

# Build feature matrix: each row = (obs1_mean, obs2_mean, ...) for one d
# Try to find a linear combination that predicts d
from numpy.linalg import lstsq

obs_names_for_combo = [k for k in observables.keys()
                       if not any(np.isnan(np.mean(results[k][d])) for d in DIMS)]

if len(obs_names_for_combo) >= 2:
    X = np.array([[np.mean(results[obs][d]) for obs in obs_names_for_combo] for d in DIMS])
    y = np.array(DIMS, dtype=float)

    # Add intercept
    X_aug = np.column_stack([X, np.ones(len(DIMS))])
    coef, residuals, rank, sv = lstsq(X_aug, y, rcond=None)

    d_pred = X_aug @ coef
    errors = np.abs(d_pred - y)
    print(f"  Features used: {obs_names_for_combo}")
    for i, d in enumerate(DIMS):
        print(f"  d={d}: predicted={d_pred[i]:.2f}, error={errors[i]:.2f}")
    print(f"  Max error: {np.max(errors):.3f}")
    print(f"  Mean error: {np.mean(errors):.3f}")

    if np.max(errors) < 0.3:
        print("  *** EXCELLENT: Combined encoder predicts dimension to within 0.3 ***")
    elif np.max(errors) < 0.5:
        print("  ** GOOD: Combined encoder predicts dimension to within 0.5 **")
else:
    print("  Insufficient clean observables for combined encoder.")


# ============================================================
# FINAL SUMMARY TABLE
# ============================================================
print("\n" + "=" * 78)
print("FINAL SUMMARY TABLE")
print("=" * 78)
print(f"{'Observable':<30} " + " ".join(f"{'d='+str(d):>10}" for d in DIMS))
print("-" * 78)

for obs_name, (label, _) in observables.items():
    short = label.split(":")[1].strip()[:28]
    row = f"{short:<30} "
    for d in DIMS:
        vals = results[obs_name][d]
        m = np.mean(vals)
        row += f"{m:>10.4f}"
    print(row)

print("\n" + "=" * 78)
print(f"EXPERIMENT COMPLETE. Total time: {time.time() - t_start:.1f}s")
print("=" * 78)

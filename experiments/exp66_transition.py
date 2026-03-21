"""
Experiment 66: BD Phase Transition Deep-Dive — Ideas 181-190

STRATEGY: Test NEW observables ACROSS the BD transition at N=50, eps=0.12.
The transition is where the physics lives. Which observables have the sharpest
jump at beta_c?

Scan beta = 0, 0.5*beta_c, beta_c, 1.5*beta_c, 2*beta_c, 3*beta_c, 5*beta_c
For each beta, run MCMC (15000 steps, 7500 thermalization), measure on 5 samples
taken every 500 steps from post-thermalization chain (reduce noise).

beta_c = 1.66 / (N * eps^2) = 1.66 / (50 * 0.0144) = 2.306

Ideas:
181. Fiedler value of Hasse Laplacian: Does it jump at beta_c?
182. Treewidth (via greedy min-degree): Does it jump at beta_c?
183. Compressibility (number of significant SVs of C): Change at beta_c?
184. Diameter of Hasse diagram: Change at beta_c?
185. Quantum walk spreading exponent: Change at beta_c?
186. Link fraction (links / total relations): Change at beta_c?
187. Hasse clique number (proxy for chromatic number): Change at beta_c?
188. Row entropy of causal matrix (causal entropy): Change at beta_c?
189. Longest antichain / sqrt(N): Change at beta_c?
190. Combined geometric fingerprint — PCA on all observables, first PC separates phases?

Reference: Interval entropy jumps 87% at beta_c (from previous experiments with
ensemble averages). We average over 5 samples per beta to reduce noise.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import linalg, stats
from causal_sets.two_orders import TwoOrder, swap_move
from causal_sets.two_orders_v2 import bd_action_corrected
from causal_sets.fast_core import FastCausalSet
from causal_sets.bd_action import count_intervals_by_size
import time

rng = np.random.default_rng(42)

# ============================================================
# PARAMETERS
# ============================================================
N = 50
eps = 0.12
beta_c = 1.66 / (N * eps**2)
beta_multiples = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
betas = [m * beta_c for m in beta_multiples]
n_steps = 15000
n_therm = 7500
record_every = 500  # record samples every 500 steps after thermalization
# This gives us (15000-7500)/500 = 15 samples per beta

print("=" * 78)
print("EXPERIMENT 66: BD PHASE TRANSITION DEEP-DIVE — IDEAS 181-190")
print("=" * 78)
print(f"N={N}, eps={eps}, beta_c={beta_c:.3f}")
print(f"Beta scan: {['%.2f (%.1f*bc)' % (b, m) for b, m in zip(betas, beta_multiples)]}")
print(f"MCMC: {n_steps} steps, {n_therm} thermalization, record every {record_every}")
print()


# ============================================================
# MCMC SAMPLER — returns list of (TwoOrder, FastCausalSet) samples
# ============================================================
def run_mcmc_samples(N, beta, eps, n_steps, n_therm, record_every, rng):
    """MCMC loop returning multiple post-thermalization samples."""
    current = TwoOrder(N, rng=rng)
    current_cs = current.to_causet()
    current_S = bd_action_corrected(current_cs, eps)
    n_acc = 0

    samples = []
    actions = []

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

        if step >= n_therm and (step - n_therm) % record_every == 0:
            samples.append((current.copy(), current_cs, current_S))
            actions.append(current_S)

    return samples, actions, n_acc / n_steps


# ============================================================
# OBSERVABLE FUNCTIONS
# ============================================================

def hasse_laplacian_fiedler(cs):
    """Idea 181: Fiedler value (2nd smallest eigenvalue) of Hasse diagram Laplacian."""
    links = cs.link_matrix()
    adj = (links | links.T).astype(float)
    degree = np.sum(adj, axis=1)
    L = np.diag(degree) - adj
    evals = np.linalg.eigvalsh(L)
    evals = np.sort(evals)
    if len(evals) >= 2:
        return evals[1]
    return 0.0


def greedy_treewidth(cs):
    """Idea 182: Treewidth via greedy min-degree elimination on comparability graph."""
    adj = cs.order | cs.order.T
    n = cs.n
    remaining = set(range(n))
    neighbors = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                neighbors[i].add(j)
                neighbors[j].add(i)

    tw = 0
    for _ in range(n):
        if not remaining:
            break
        min_v = min(remaining, key=lambda v: len(neighbors[v] & remaining))
        nbrs = neighbors[min_v] & remaining
        tw = max(tw, len(nbrs))
        nbrs_list = list(nbrs)
        for a in range(len(nbrs_list)):
            for b in range(a+1, len(nbrs_list)):
                u, v = nbrs_list[a], nbrs_list[b]
                neighbors[u].add(v)
                neighbors[v].add(u)
        remaining.remove(min_v)
    return tw


def sv_compressibility(cs):
    """Idea 183: Number of significant singular values of C (> 1% of max)."""
    C = cs.order.astype(float)
    svs = np.linalg.svd(C, compute_uv=False)
    threshold = 0.01 * svs[0]
    return int(np.sum(svs > threshold))


def hasse_diameter(cs):
    """Idea 184: Diameter of the Hasse diagram (longest shortest path in link graph)."""
    links = cs.link_matrix()
    adj = (links | links.T).astype(np.int32)
    n = cs.n
    max_dist = 0
    for start in range(n):
        dist = -np.ones(n, dtype=int)
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            v = queue[head]
            head += 1
            for u in range(n):
                if adj[v, u] and dist[u] == -1:
                    dist[u] = dist[v] + 1
                    queue.append(u)
        reachable = dist[dist >= 0]
        if len(reachable) > 0:
            max_dist = max(max_dist, np.max(reachable))
    return max_dist


def quantum_walk_spread(cs):
    """Idea 185: Quantum walk spreading on Hasse DAG.
    Variance of continuous-time quantum walk at t=5."""
    links = cs.link_matrix()
    adj = (links | links.T).astype(float)
    n = cs.n
    degree = np.sum(adj, axis=1)
    active = np.where(degree > 0)[0]
    if len(active) < 3:
        return 0.0
    adj_sub = adj[np.ix_(active, active)]
    n_sub = len(active)

    psi0 = np.zeros(n_sub, dtype=complex)
    psi0[n_sub // 2] = 1.0

    H = adj_sub
    evals, evecs = np.linalg.eigh(H)

    t = 5.0
    phases = np.exp(-1j * evals * t)
    psi_t = evecs @ (phases * (evecs.T @ psi0))
    prob = np.abs(psi_t)**2

    positions = np.arange(n_sub, dtype=float)
    mean_pos = np.sum(positions * prob)
    var_pos = np.sum((positions - mean_pos)**2 * prob)
    return var_pos


def link_fraction(cs):
    """Idea 186: links / total relations."""
    links = cs.link_matrix()
    n_links = int(np.sum(links))
    n_rels = cs.num_relations()
    if n_rels == 0:
        return 1.0
    return n_links / n_rels


def hasse_clique_number(cs):
    """Idea 187: Clique number of undirected Hasse (link) graph. Greedy."""
    links = cs.link_matrix()
    adj = (links | links.T).astype(bool)
    n = cs.n
    best_clique = 0
    for start in range(n):
        clique = {start}
        candidates = set(np.where(adj[start])[0])
        while candidates:
            best_v = None
            best_deg = -1
            for v in candidates:
                if all(adj[v, c] for c in clique):
                    deg = np.sum(adj[v])
                    if deg > best_deg:
                        best_v = v
                        best_deg = deg
            if best_v is None:
                break
            clique.add(best_v)
            candidates = candidates & set(np.where(adj[best_v])[0])
        best_clique = max(best_clique, len(clique))
    return best_clique


def row_entropy(cs):
    """Idea 188: Mean row entropy of causal matrix."""
    C = cs.order.astype(float)
    n = cs.n
    entropies = []
    for i in range(n):
        row = C[i, :]
        total = np.sum(row)
        if total < 1:
            entropies.append(0.0)
            continue
        p = row / total
        p = p[p > 0]
        entropies.append(-np.sum(p * np.log2(p)))
    return np.mean(entropies)


def longest_antichain_normalized(two_order):
    """Idea 189: Longest antichain / sqrt(N). Via LDS of v*u^{-1}."""
    N = two_order.N
    u_inv = np.argsort(two_order.u)
    pi = two_order.v[u_inv]
    # LDS(pi) = LIS(reversed(pi))
    rev_pi = pi[::-1]
    tails = []
    for x in rev_pi:
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(tails):
            tails.append(x)
        else:
            tails[lo] = x
    return len(tails) / np.sqrt(N)


def interval_entropy(cs):
    """Reference: interval entropy (entropy of interval size distribution)."""
    pairs, sizes = cs.interval_sizes_vectorized()
    if len(sizes) == 0:
        return 0.0
    unique, counts = np.unique(sizes, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p))


def ordering_fraction(cs):
    """Extra reference: ordering fraction = relations / (N choose 2)."""
    return cs.ordering_fraction()


# ============================================================
# RUN MCMC AT EACH BETA AND MEASURE ALL OBSERVABLES (AVERAGED)
# ============================================================

print("Running MCMC at each beta value, averaging over post-thermalization samples...")
print("-" * 78)

obs_names = [
    "181:Fiedler", "182:Treewidth", "183:SVCompress", "184:Diameter",
    "185:QWalkVar", "186:LinkFrac", "187:CliqueNum", "188:RowEntropy",
    "189:AC/sqrtN", "REF:IntEntropy", "REF:OrdFrac", "REF:Action"
]

all_means = {name: [] for name in obs_names}
all_stds = {name: [] for name in obs_names}

for mult, beta in zip(beta_multiples, betas):
    t0 = time.time()
    print(f"\nbeta = {beta:.3f} ({mult:.1f} * beta_c):", flush=True)

    samples, actions_list, acc = run_mcmc_samples(
        N, beta, eps, n_steps, n_therm, record_every, rng)

    dt_mcmc = time.time() - t0
    n_samples = len(samples)
    mean_action = np.mean(actions_list)
    print(f"  MCMC: {dt_mcmc:.1f}s, accept={acc:.3f}, <S>={mean_action:.3f}, "
          f"<S>/N={mean_action/N:.4f}, {n_samples} samples")

    t1 = time.time()

    # Measure all observables on each sample, then average
    obs_vals = {name: [] for name in obs_names}

    for s_idx, (two_order, cs, act) in enumerate(samples):
        obs_vals["181:Fiedler"].append(hasse_laplacian_fiedler(cs))
        obs_vals["182:Treewidth"].append(greedy_treewidth(cs))
        obs_vals["183:SVCompress"].append(sv_compressibility(cs))
        obs_vals["184:Diameter"].append(hasse_diameter(cs))
        obs_vals["185:QWalkVar"].append(quantum_walk_spread(cs))
        obs_vals["186:LinkFrac"].append(link_fraction(cs))
        obs_vals["187:CliqueNum"].append(hasse_clique_number(cs))
        obs_vals["188:RowEntropy"].append(row_entropy(cs))
        obs_vals["189:AC/sqrtN"].append(longest_antichain_normalized(two_order))
        obs_vals["REF:IntEntropy"].append(interval_entropy(cs))
        obs_vals["REF:OrdFrac"].append(ordering_fraction(cs))
        obs_vals["REF:Action"].append(act)

    dt_obs = time.time() - t1
    print(f"  Observables on {n_samples} samples in {dt_obs:.1f}s")

    for name in obs_names:
        vals = obs_vals[name]
        m = np.mean(vals)
        s = np.std(vals)
        all_means[name].append(m)
        all_stds[name].append(s)
        print(f"    {name:20s} = {m:>10.4f} +/- {s:>8.4f}")


# ============================================================
# ANALYSIS: Jump magnitudes at beta_c
# ============================================================

print("\n" + "=" * 78)
print("ANALYSIS: OBSERVABLE JUMPS ACROSS THE PHASE TRANSITION")
print("=" * 78)

# Show key beta values
print(f"\n{'Observable':22s} {'b=0':>10s} {'0.5bc':>10s} {'bc':>10s} "
      f"{'1.5bc':>10s} {'2bc':>10s} {'3bc':>10s} {'5bc':>10s}")
print("-" * 92)
for name in obs_names:
    vals = all_means[name]
    print(f"{name:22s}", end="")
    for v in vals:
        print(f" {v:>10.4f}", end="")
    print()

# Compute relative jump: |value(5*beta_c) - value(0)| / |value(0)|
# This is the "total" jump from disordered to deep ordered phase
print(f"\n{'Observable':22s} {'Jump(0->5bc)%':>14s} {'Jump(0->2bc)%':>14s} {'Monotonic':>10s} {'Direction':>10s}")
print("-" * 74)

jumps = {}
jump_0_2bc = {}
for name in obs_names:
    vals = all_means[name]
    v0 = vals[0]       # beta=0
    v2c = vals[4]      # beta=2*beta_c
    v5c = vals[6]      # beta=5*beta_c

    if abs(v0) > 1e-10:
        jump_total = abs(v5c - v0) / abs(v0) * 100
        jump_2bc = abs(v2c - v0) / abs(v0) * 100
    else:
        jump_total = abs(v5c - v0) * 100
        jump_2bc = abs(v2c - v0) * 100

    # Check monotonicity (trend direction)
    diffs = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
    n_pos = sum(1 for d in diffs if d > 0)
    n_neg = sum(1 for d in diffs if d < 0)
    if n_pos >= 5:
        monotonic = "~yes"
        direction = "UP"
    elif n_neg >= 5:
        monotonic = "~yes"
        direction = "DOWN"
    else:
        monotonic = "no"
        direction = "mixed"

    jumps[name] = jump_total
    jump_0_2bc[name] = jump_2bc
    print(f"{name:22s} {jump_total:>13.1f}% {jump_2bc:>13.1f}% {monotonic:>10s} {direction:>10s}")


# ============================================================
# IDEA 190: PCA FINGERPRINT
# ============================================================

print("\n" + "=" * 78)
print("IDEA 190: PCA GEOMETRIC FINGERPRINT")
print("=" * 78)

# Build feature matrix: 7 beta values x 9 observables (excluding references)
feature_names = [n for n in obs_names if not n.startswith("REF:")]
X = np.array([[all_means[name][i] for name in feature_names] for i in range(len(betas))])

# Standardize
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std < 1e-10] = 1.0
X_norm = (X - X_mean) / X_std

# PCA
cov = np.cov(X_norm.T)
evals_pca, evecs_pca = np.linalg.eigh(cov)
idx = np.argsort(evals_pca)[::-1]
evals_pca = evals_pca[idx]
evecs_pca = evecs_pca[:, idx]

var_explained = evals_pca / evals_pca.sum() * 100

print(f"\nPCA variance explained:")
for i in range(min(5, len(evals_pca))):
    print(f"  PC{i+1}: {var_explained[i]:.1f}%")
print(f"  PC1+PC2: {var_explained[0]+var_explained[1]:.1f}%")

# PC1 scores
pc1_scores = X_norm @ evecs_pca[:, 0]
print(f"\nPC1 scores across the transition:")
for i, (mult, beta) in enumerate(zip(beta_multiples, betas)):
    label = "DISORDERED" if mult < 1.0 else ("CRITICAL" if mult == 1.0 else "ORDERED")
    print(f"  beta/bc={mult:.1f} [{label:>10s}]: PC1 = {pc1_scores[i]:+.3f}")

# PC1 loadings
print(f"\nPC1 loadings (which observables drive phase separation?):")
for j, name in enumerate(feature_names):
    print(f"  {name:20s}: {evecs_pca[j, 0]:+.4f}")

# Phase separation analysis
disordered_pc1 = pc1_scores[:2]  # beta=0, 0.5*beta_c
ordered_pc1 = pc1_scores[4:]     # 2*beta_c, 3*beta_c, 5*beta_c

d_mean = np.mean(disordered_pc1)
o_mean = np.mean(ordered_pc1)
# Gap: minimum distance between disordered and ordered PC1 scores
if o_mean > d_mean:
    sep = np.min(ordered_pc1) - np.max(disordered_pc1)
else:
    sep = np.min(disordered_pc1) - np.max(ordered_pc1)

print(f"\n  Disordered mean PC1: {d_mean:+.3f}")
print(f"  Ordered mean PC1:    {o_mean:+.3f}")
print(f"  Separation gap:      {sep:+.3f}")
print(f"  Clean separation?    {'YES' if sep > 0 else 'NO (overlap)'}")


# ============================================================
# RANKING AND SCORING
# ============================================================

print("\n" + "=" * 78)
print("RANKING: SHARPEST JUMPS AT THE PHASE TRANSITION")
print("=" * 78)

ref_jump = jumps.get("REF:IntEntropy", 0)
ref_jump_ord = jumps.get("REF:OrdFrac", 0)
print(f"\nReference jumps (0 -> 5*beta_c):")
print(f"  Interval entropy: {ref_jump:.1f}%")
print(f"  Ordering fraction: {ref_jump_ord:.1f}%")
print(f"  Action: {jumps.get('REF:Action', 0):.1f}%")

# Rank all non-reference observables
non_ref = [(name, jumps[name]) for name in obs_names if not name.startswith("REF:")]
ranked = sorted(non_ref, key=lambda x: x[1], reverse=True)

print(f"\n{'Rank':>4s} {'Observable':22s} {'Jump(0->5bc)':>14s} {'vs OrdFrac':>12s}")
print("-" * 56)
for rank, (name, jump) in enumerate(ranked, 1):
    vs_ref = f"{jump/ref_jump_ord:.2f}x" if ref_jump_ord > 0 else "N/A"
    marker = " ***" if jump > ref_jump_ord else ""
    print(f"{rank:>4d} {name:22s} {jump:>13.1f}% {vs_ref:>12s}{marker}")


# ============================================================
# SCORE EACH IDEA
# ============================================================

print("\n" + "=" * 78)
print("SCORES AND VERDICTS FOR IDEAS 181-190")
print("=" * 78)

verdicts = [
    ("181", "181:Fiedler", "Fiedler value of Hasse Laplacian",
     "Algebraic connectivity of the link graph. Measures how well-connected the "
     "Hasse diagram is."),
    ("182", "182:Treewidth", "Treewidth of comparability graph",
     "Structural complexity. Low treewidth = tree-like; high = complex graph."),
    ("183", "183:SVCompress", "SV Compressibility of C",
     "Number of significant singular values. Low = simple/compressible structure."),
    ("184", "184:Diameter", "Hasse diagram diameter",
     "Longest shortest path in the link graph. Probes geometric extent."),
    ("185", "185:QWalkVar", "Quantum walk spreading variance",
     "Continuous-time quantum walk variance. Probes Lorentzian transport."),
    ("186", "186:LinkFrac", "Link fraction (links / relations)",
     "Sparsity of the Hasse diagram relative to full order."),
    ("187", "187:CliqueNum", "Hasse clique number",
     "Largest clique in the link graph. Probes local clustering."),
    ("188", "188:RowEntropy", "Row entropy of causal matrix",
     "Mean Shannon entropy per row. Measures causal uniformity."),
    ("189", "189:AC/sqrtN", "Longest antichain / sqrt(N)",
     "Normalized width. Connects to Vershik-Kerov (Idea 139)."),
    ("190", None, "PCA geometric fingerprint",
     "Combined multi-observable order parameter via PCA."),
]

for idea_num, key, title, description in verdicts:
    if key and key in jumps:
        j = jumps[key]
        vals = all_means[key]
        # Check trend quality
        diffs = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
        n_consistent = max(sum(1 for d in diffs if d > 0), sum(1 for d in diffs if d < 0))
        trend_quality = n_consistent / len(diffs)

        if j > 50 and trend_quality > 0.7:
            score = 7.5
        elif j > 30 and trend_quality > 0.6:
            score = 7.0
        elif j > 15:
            score = 6.5
        elif j > 5:
            score = 6.0
        elif j > 2:
            score = 5.5
        else:
            score = 5.0
    elif idea_num == "190":
        j = 0
        if var_explained[0] > 70 and sep > 0:
            score = 7.0
        elif var_explained[0] > 50:
            score = 6.0
        else:
            score = 5.5
    else:
        j = 0
        score = 5.0

    print(f"\n  Idea {idea_num}: {title}")
    print(f"    {description}")
    if key and key in jumps:
        print(f"    Jump: {j:.1f}%  |  Trend quality: {trend_quality:.0%}")
        print(f"    Values: {' -> '.join(['%.3f' % v for v in all_means[key]])}")
    elif idea_num == "190":
        print(f"    PC1 explains {var_explained[0]:.0f}% of variance")
        print(f"    Phase separation gap: {sep:+.3f} ({'clean' if sep > 0 else 'overlapping'})")
    print(f"    Score: {score}/10")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("FINAL SUMMARY")
print("=" * 78)

print(f"""
Parameters: N={N}, eps={eps}, beta_c={beta_c:.3f}
Beta scan: {len(betas)} values from 0 to {betas[-1]:.2f} ({beta_multiples[-1]}x beta_c)
MCMC: {n_steps} steps, {n_therm} thermalization, ~{len(samples)} samples per beta

Key results:
  - Action <S>/N:    {all_means['REF:Action'][0]/N:.4f} (beta=0) -> {all_means['REF:Action'][-1]/N:.4f} (5*beta_c)
  - Ordering frac:   {all_means['REF:OrdFrac'][0]:.4f} (beta=0) -> {all_means['REF:OrdFrac'][-1]:.4f} (5*beta_c)

Top 3 sharpest NEW observables (0 -> 5*beta_c):
""")

for rank, (name, jump) in enumerate(ranked[:3], 1):
    print(f"  {rank}. {name}: {jump:.1f}% jump")

beat_ord = [name for name, jump in ranked if jump > ref_jump_ord]
if beat_ord:
    print(f"\n  Observables with larger jump than ordering fraction ({ref_jump_ord:.1f}%):")
    for name in beat_ord:
        print(f"    - {name}: {jumps[name]:.1f}%")

print(f"""
PCA fingerprint: PC1 explains {var_explained[0]:.1f}% of variance
  Phase separation: {'YES' if sep > 0 else 'NO'} (gap = {sep:.3f})

Key physical insight:
  The observables that change most across the transition tell us what
  structurally distinguishes the ordered (manifold-like) phase from
  the disordered phase.

190 total ideas tested across the quantum gravity research program.
""")

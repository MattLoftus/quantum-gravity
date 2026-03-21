"""
Experiment 68, Round 11: Ideas 201-210 — PURE CAUSAL GEOMETRY

Focus: Properties of the partial order itself, no SJ vacuum.
Best from round 2: Fiedler value (7.5), link fraction (7.5), treewidth (7.0), antichain theorem (7.0).

Ideas:
  201. Graph homomorphism density: count homomorphisms from small posets (2-chain, V, Λ, diamond)
       into the causet. Ratios of these counts characterize dimension.
  202. Order-theoretic width function: width(k) = max antichain in elements at "height" k.
       Width profile shape distinguishes manifold-like from random causets.
  203. Ramsey numbers of causal sets: min N such that every 2-order on N has a chain of
       length a or an antichain of length b. Compare to Dilworth/Ramsey bounds.
  204. Forbidden subposet characterization: frequency of "N-poset" (incomparable pair with
       common past and future) vs "Z-poset" in 2D vs higher-d causets.
  205. Causal set automorphism group size: |Aut(C)| for random 2-orders vs sprinkled causets.
       Manifold-like causets should have ~trivial automorphisms.
  206. Order polytope volume: the order polytope P(C) ⊂ R^N has volume = 1/(#linear extensions).
       Estimate via sampling; relates to entropy of the causal order.
  207. Chain decomposition structure: Dilworth's theorem gives min # chains covering the poset.
       Distribution of chain lengths in the optimal decomposition.
  208. Path enumeration: count directed paths of each length in the Hasse diagram (link graph).
       Path length distribution as a dimension estimator.
  209. Möbius function statistics: the Möbius function μ(x,y) of the poset encodes inclusion-
       exclusion. Distribution of μ values for random 2-orders vs sprinkled causets.
  210. Convex subposet density: fraction of k-element subsets that form convex sub-causets
       (i.e., if x,y in S and x<z<y then z in S). Measures "geodesic completeness."
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats
from itertools import permutations, combinations
from collections import Counter, defaultdict
from causal_sets.two_orders import TwoOrder, swap_move
from causal_sets.fast_core import FastCausalSet, sprinkle_fast
import time

np.set_printoptions(precision=6, suppress=True, linewidth=120)
rng = np.random.default_rng(42)


# ============================================================
# SHARED UTILITIES
# ============================================================

def two_order_from_perms(u, v):
    """Create a FastCausalSet from two permutations."""
    N = len(u)
    cs = FastCausalSet(N)
    u_arr = np.array(u)
    v_arr = np.array(v)
    cs.order = (u_arr[:, None] < u_arr[None, :]) & (v_arr[:, None] < v_arr[None, :])
    return cs

def make_2order_causet(N, rng):
    to = TwoOrder(N, rng=rng)
    return to, to.to_causet()

def random_dag(N, density, rng):
    """Random DAG with given density (transitively closed)."""
    cs = FastCausalSet(N)
    mask = rng.random((N, N)) < density
    cs.order = np.triu(mask, k=1)
    order_int = cs.order.astype(np.int32)
    changed = True
    while changed:
        new_order = (order_int @ order_int > 0) | cs.order
        changed = np.any(new_order != cs.order)
        cs.order = new_order
        order_int = cs.order.astype(np.int32)
    return cs

def random_poset(N, p, rng):
    """Random poset: random DAG with edge probability p, then transitive closure."""
    return random_dag(N, p, rng)

def fiedler_value(cs):
    """Second-smallest eigenvalue of the link graph Laplacian."""
    links = cs.link_matrix()
    adj = links | links.T
    degree = np.sum(adj, axis=1).astype(float)
    mask = degree > 0
    adj_sub = adj[np.ix_(mask, mask)].astype(float)
    degree_sub = degree[mask]
    n = adj_sub.shape[0]
    if n < 3:
        return 0.0
    L = np.diag(degree_sub) - adj_sub
    evals = np.linalg.eigvalsh(L)
    evals = np.sort(evals)
    return float(evals[1]) if len(evals) > 1 else 0.0

def longest_antichain_size(cs):
    """Max antichain via bipartite matching (Dilworth)."""
    N = cs.n
    order = cs.order
    matched_right = [-1] * N
    def dfs(u, visited):
        for v in range(N):
            if order[u, v] and not visited[v]:
                visited[v] = True
                if matched_right[v] == -1 or dfs(matched_right[v], visited):
                    matched_right[v] = u
                    return True
        return False
    matching = 0
    for u in range(N):
        visited = [False] * N
        if dfs(u, visited):
            matching += 1
    return N - matching

def element_heights(cs):
    """Compute height of each element (length of longest chain ending at it)."""
    N = cs.n
    dp = np.zeros(N, dtype=int)
    for j in range(N):
        preds = np.where(cs.order[:, j])[0]
        if len(preds) > 0:
            dp[j] = np.max(dp[preds]) + 1
    return dp


print("=" * 78)
print("EXPERIMENT 68, ROUND 11: PURE CAUSAL GEOMETRY (Ideas 201-210)")
print("=" * 78)


# ============================================================
# IDEA 201: GRAPH HOMOMORPHISM DENSITY
# ============================================================
print("\n" + "=" * 78)
print("IDEA 201: Subposet pattern densities (homomorphism counts)")
print("Count occurrences of small posets: 2-chain, V (fork), Λ (join), diamond, N-poset")
print("=" * 78)

def count_pattern_2chain(order):
    """Count 2-chains: i < j < k."""
    # order @ order gives paths of length 2
    o = order.astype(np.int32)
    return int(np.sum(o @ o))

def count_pattern_v(order):
    """Count V-patterns (forks): j > i and j > k, i incomparable to k."""
    N = order.shape[0]
    # For each j, count pairs (i,k) where order[i,j]=True and order[k,j]=True and NOT order[i,k] and NOT order[k,i]
    count = 0
    for j in range(N):
        pasts = np.where(order[:, j])[0]
        npast = len(pasts)
        if npast < 2:
            continue
        for a in range(npast):
            for b in range(a+1, npast):
                i, k = pasts[a], pasts[b]
                if not order[i, k] and not order[k, i]:
                    count += 1
    return count

def count_pattern_lambda(order):
    """Count Λ-patterns (joins): i < j and i < k, j incomparable to k."""
    N = order.shape[0]
    count = 0
    for i in range(N):
        futures = np.where(order[i, :])[0]
        nfut = len(futures)
        if nfut < 2:
            continue
        for a in range(nfut):
            for b in range(a+1, nfut):
                j, k = futures[a], futures[b]
                if not order[j, k] and not order[k, j]:
                    count += 1
    return count

def count_pattern_diamond(order):
    """Count diamonds: i < j, i < k, j < m, k < m, j incomp k."""
    N = order.shape[0]
    count = 0
    for i in range(N):
        futures_i = set(np.where(order[i, :])[0])
        for m in range(N):
            if m == i or not order[i, m]:
                continue
            pasts_m = set(np.where(order[:, m])[0])
            # middle elements: in future of i AND past of m, not i or m
            middle = (futures_i & pasts_m) - {i, m}
            middle = list(middle)
            nmid = len(middle)
            if nmid < 2:
                continue
            for a in range(nmid):
                for b in range(a+1, nmid):
                    j, k = middle[a], middle[b]
                    if not order[j, k] and not order[k, j]:
                        count += 1
    return count

t0 = time.time()
N_test = 30
n_trials = 40

print(f"\nComparing pattern densities: 2-orders (N={N_test}) vs random DAGs vs sprinkled 2D")
print(f"  {n_trials} trials each\n")

results_201 = {}
for label, gen_func in [
    ("2-order", lambda: make_2order_causet(N_test, rng)[1]),
    ("random DAG (p=0.15)", lambda: random_poset(N_test, 0.15, rng)),
    ("sprinkled 2D", lambda: sprinkle_fast(N_test, dim=2, rng=rng)[0]),
]:
    chains, vs, lambdas, diamonds = [], [], [], []
    for _ in range(n_trials):
        cs = gen_func()
        o = cs.order
        chains.append(count_pattern_2chain(o))
        vs.append(count_pattern_v(o))
        lambdas.append(count_pattern_lambda(o))
        diamonds.append(count_pattern_diamond(o))

    # Normalize by number of ordered triples possible
    n_pairs = N_test * (N_test - 1) * (N_test - 2) / 6
    results_201[label] = {
        '2-chain': np.mean(chains),
        'V-fork': np.mean(vs),
        'Λ-join': np.mean(lambdas),
        'diamond': np.mean(diamonds),
    }
    print(f"  {label}:")
    print(f"    2-chain: {np.mean(chains):.1f} ± {np.std(chains):.1f}")
    print(f"    V-fork:  {np.mean(vs):.1f} ± {np.std(vs):.1f}")
    print(f"    Λ-join:  {np.mean(lambdas):.1f} ± {np.std(lambdas):.1f}")
    print(f"    diamond: {np.mean(diamonds):.1f} ± {np.std(diamonds):.1f}")

# Compute ratios
print("\n  Pattern RATIOS (V-fork / 2-chain, diamond / 2-chain):")
for label in results_201:
    r = results_201[label]
    v_ratio = r['V-fork'] / max(r['2-chain'], 1)
    d_ratio = r['diamond'] / max(r['2-chain'], 1)
    lam_ratio = r['Λ-join'] / max(r['2-chain'], 1)
    print(f"    {label}: V/chain={v_ratio:.3f}, Λ/chain={lam_ratio:.3f}, diamond/chain={d_ratio:.4f}")

print(f"\n  Time: {time.time() - t0:.1f}s")
print("\n  VERDICT: Pattern densities quantify the 'shape' of the partial order.")
print("  If V/chain and Λ/chain ratios differ significantly between 2-orders and")
print("  random DAGs, this is a genuine structural signature.")


# ============================================================
# IDEA 202: WIDTH PROFILE
# ============================================================
print("\n" + "=" * 78)
print("IDEA 202: Width profile — antichain size at each height level")
print("=" * 78)

def width_profile(cs):
    """Compute width at each height level: number of elements at height k."""
    heights = element_heights(cs)
    max_h = int(np.max(heights)) if len(heights) > 0 else 0
    profile = np.zeros(max_h + 1, dtype=int)
    for h in heights:
        profile[h] += 1
    return profile

def width_profile_stats(cs):
    """Summary statistics of the width profile."""
    profile = width_profile(cs)
    if len(profile) == 0:
        return 0, 0, 0
    max_width = int(np.max(profile))
    mean_width = float(np.mean(profile))
    # Symmetry: compare first half to reversed second half
    n = len(profile)
    if n < 3:
        return max_width, mean_width, 0.0
    first_half = profile[:n//2]
    second_half = profile[(n+1)//2:][::-1]
    min_len = min(len(first_half), len(second_half))
    if min_len == 0:
        return max_width, mean_width, 0.0
    corr = np.corrcoef(first_half[:min_len].astype(float),
                         second_half[:min_len].astype(float))[0, 1]
    return max_width, mean_width, corr if not np.isnan(corr) else 0.0

t0 = time.time()
N_test = 60
n_trials = 50

print(f"\nWidth profile stats: 2-orders vs random DAGs vs sprinkled 2D (N={N_test})")

for label, gen_func in [
    ("2-order", lambda: make_2order_causet(N_test, rng)[1]),
    ("random DAG (p=0.15)", lambda: random_poset(N_test, 0.15, rng)),
    ("sprinkled 2D", lambda: sprinkle_fast(N_test, dim=2, rng=rng)[0]),
    ("sprinkled 3D", lambda: sprinkle_fast(N_test, dim=3, rng=rng)[0]),
]:
    max_ws, mean_ws, sym_ws = [], [], []
    heights_all = []
    for _ in range(n_trials):
        cs = gen_func()
        mw, meanw, sym = width_profile_stats(cs)
        max_ws.append(mw)
        mean_ws.append(meanw)
        sym_ws.append(sym)
        heights_all.append(cs.longest_chain())

    print(f"\n  {label}:")
    print(f"    Max width:     {np.mean(max_ws):.2f} ± {np.std(max_ws):.2f}")
    print(f"    Mean width:    {np.mean(mean_ws):.2f} ± {np.std(mean_ws):.2f}")
    print(f"    Height:        {np.mean(heights_all):.2f} ± {np.std(heights_all):.2f}")
    print(f"    Width/Height:  {np.mean(np.array(max_ws)/np.maximum(heights_all,1)):.3f}")
    print(f"    Symmetry corr: {np.mean(sym_ws):.3f} ± {np.std(sym_ws):.3f}")

print(f"\n  Time: {time.time() - t0:.1f}s")
print("\n  VERDICT: Width profile captures the 'bulk geometry' shape.")
print("  Manifold-like causets should have diamond-shaped profiles (wide middle).")
print("  Random DAGs should have monotonically decreasing or flat profiles.")


# ============================================================
# IDEA 203: RAMSEY-TYPE BOUNDS FOR 2-ORDERS
# ============================================================
print("\n" + "=" * 78)
print("IDEA 203: Ramsey-type bounds — chain/antichain tradeoff")
print("The Erdős–Szekeres theorem: any sequence of > (a-1)(b-1) has monotone")
print("subsequence of length a or b. For 2-orders: chain≥a or antichain≥b.")
print("=" * 78)

t0 = time.time()
N_vals = [20, 30, 40, 50, 60, 80]
n_trials = 40

print(f"\n  Measuring chain * antichain product vs N for random 2-orders")
print(f"  Erdős–Szekeres says chain * antichain ≥ N (actually chain + antichain ≥ 1 + √N)")

for N in N_vals:
    chains, antichains, products = [], [], []
    for _ in range(n_trials):
        _, cs = make_2order_causet(N, rng)
        c = cs.longest_chain()
        a = longest_antichain_size(cs)
        chains.append(c)
        antichains.append(a)
        products.append(c * a)

    print(f"  N={N:3d}: chain={np.mean(chains):.1f}±{np.std(chains):.1f}, "
          f"antichain={np.mean(antichains):.1f}±{np.std(antichains):.1f}, "
          f"c*a={np.mean(products):.1f} (√N={np.sqrt(N):.2f}, N={N})")

# Compare to random DAGs
print(f"\n  Random DAGs for comparison:")
for N in [30, 50, 80]:
    chains, antichains, products = [], [], []
    for _ in range(n_trials):
        cs = random_poset(N, 0.15, rng)
        c = cs.longest_chain()
        a = longest_antichain_size(cs)
        chains.append(c)
        antichains.append(a)
        products.append(c * a)

    print(f"  N={N:3d}: chain={np.mean(chains):.1f}±{np.std(chains):.1f}, "
          f"antichain={np.mean(antichains):.1f}±{np.std(antichains):.1f}, "
          f"c*a={np.mean(products):.1f}")

print(f"\n  Time: {time.time() - t0:.1f}s")
print("\n  VERDICT: For 2-orders, chain ~ √N and antichain ~ √N (both scale as √N).")
print("  The product c*a should be ≈ N. This is the Vershik-Kerov / Baik-Deift-Johansson")
print("  result for longest increasing subsequence. Compare DAG behavior.")


# ============================================================
# IDEA 204: FORBIDDEN SUBPOSET FREQUENCIES (N-poset, Z-poset)
# ============================================================
print("\n" + "=" * 78)
print("IDEA 204: Forbidden subposet frequencies — N-poset and Z-poset")
print("The N-poset: a<c, b<d, a<d (but a,b incomp and c,d incomp).")
print("The Z-poset: a<b, c<d, a<d (but a,c incomp and b,d incomp).")
print("=" * 78)

def count_n_poset(order):
    """Count N-posets: a<c, b<d, a<d, but {a,b} incomparable and {c,d} incomparable."""
    N = order.shape[0]
    count = 0
    # For efficiency, iterate over (a,d) pairs where a<d
    related = np.where(np.triu(order, k=1))
    for idx in range(len(related[0])):
        a, d = related[0][idx], related[1][idx]
        # Find c: a<c, c incomp d
        futures_a = np.where(order[a, :])[0]
        for c in futures_a:
            if c == d or order[c, d] or order[d, c]:
                continue
            # Find b: b<d, b incomp a
            pasts_d = np.where(order[:, d])[0]
            for b in pasts_d:
                if b == a or b == c or order[a, b] or order[b, a]:
                    continue
                count += 1
    return count

def count_z_poset(order):
    """Count Z-posets: a<b, c<d, a<d, but {a,c} incomp and {b,d} incomp."""
    N = order.shape[0]
    count = 0
    related = np.where(np.triu(order, k=1))
    for idx in range(len(related[0])):
        a, d = related[0][idx], related[1][idx]
        # Find b: a<b, b incomp d
        futures_a = np.where(order[a, :])[0]
        for b in futures_a:
            if b == d or order[b, d] or order[d, b]:
                continue
            # Find c: c<d, c incomp a
            pasts_d = np.where(order[:, d])[0]
            for c in pasts_d:
                if c == a or c == b or order[a, c] or order[c, a]:
                    continue
                count += 1
    return count

t0 = time.time()
N_test = 25  # keep small — O(N^4) counting
n_trials = 30

print(f"\n  Counting N-posets and Z-posets (N={N_test}, {n_trials} trials)")

for label, gen_func in [
    ("2-order", lambda: make_2order_causet(N_test, rng)[1]),
    ("random DAG (p=0.15)", lambda: random_poset(N_test, 0.15, rng)),
    ("sprinkled 2D", lambda: sprinkle_fast(N_test, dim=2, rng=rng)[0]),
]:
    n_counts, z_counts, ratios = [], [], []
    for _ in range(n_trials):
        cs = gen_func()
        n_ct = count_n_poset(cs.order)
        z_ct = count_z_poset(cs.order)
        n_counts.append(n_ct)
        z_counts.append(z_ct)
        ratios.append(n_ct / max(z_ct, 1))

    print(f"\n  {label}:")
    print(f"    N-poset count: {np.mean(n_counts):.1f} ± {np.std(n_counts):.1f}")
    print(f"    Z-poset count: {np.mean(z_counts):.1f} ± {np.std(z_counts):.1f}")
    print(f"    N/Z ratio:     {np.mean(ratios):.3f} ± {np.std(ratios):.3f}")

print(f"\n  Time: {time.time() - t0:.1f}s")
print("\n  VERDICT: If N/Z ratio differs between 2-orders and random DAGs,")
print("  it characterizes the specific correlations imposed by 2D embeddability.")


# ============================================================
# IDEA 205: AUTOMORPHISM GROUP SIZE
# ============================================================
print("\n" + "=" * 78)
print("IDEA 205: Automorphism group size")
print("Manifold-like causets should be rigid (trivial automorphism group).")
print("=" * 78)

def count_automorphisms_small(order):
    """Brute-force automorphism count for small N (N ≤ 10)."""
    N = order.shape[0]
    if N > 10:
        # Use a quick heuristic: check if any transposition is an automorphism
        count = 1  # identity
        for i in range(N):
            for j in range(i+1, N):
                # Check if swapping i,j preserves order
                is_auto = True
                for a in range(N):
                    for b in range(N):
                        pa = j if a == i else (i if a == j else a)
                        pb = j if b == i else (i if b == j else b)
                        if order[a, b] != order[pa, pb]:
                            is_auto = False
                            break
                    if not is_auto:
                        break
                if is_auto:
                    count += 1
        return count

    # Full brute force for small N
    count = 0
    from itertools import permutations as perms
    for perm in perms(range(N)):
        perm = list(perm)
        is_auto = True
        for a in range(N):
            if not is_auto:
                break
            for b in range(N):
                if order[a, b] != order[perm[a], perm[b]]:
                    is_auto = False
                    break
        if is_auto:
            count += 1
    return count

t0 = time.time()

# Small N for exact computation
for N_test in [6, 7, 8]:
    n_trials = 50
    auto_2order = []
    auto_random = []

    for _ in range(n_trials):
        _, cs = make_2order_causet(N_test, rng)
        auto_2order.append(count_automorphisms_small(cs.order))

        cs_r = random_poset(N_test, 0.25, rng)
        auto_random.append(count_automorphisms_small(cs_r.order))

    print(f"\n  N={N_test}:")
    print(f"    2-order:  |Aut| = {np.mean(auto_2order):.2f} ± {np.std(auto_2order):.2f}, "
          f"trivial fraction = {np.mean(np.array(auto_2order)==1):.2f}")
    print(f"    random:   |Aut| = {np.mean(auto_random):.2f} ± {np.std(auto_random):.2f}, "
          f"trivial fraction = {np.mean(np.array(auto_random)==1):.2f}")

# Larger N: just check transpositions
N_test = 30
n_trials = 40
print(f"\n  N={N_test} (transposition automorphisms only):")
for label, gen_func in [
    ("2-order", lambda: make_2order_causet(N_test, rng)[1]),
    ("random DAG (p=0.15)", lambda: random_poset(N_test, 0.15, rng)),
    ("sprinkled 2D", lambda: sprinkle_fast(N_test, dim=2, rng=rng)[0]),
]:
    trans_autos = []
    for _ in range(n_trials):
        cs = gen_func()
        # Count transposition automorphisms
        ta = 0
        for i in range(cs.n):
            for j in range(i+1, cs.n):
                # Elements i,j swappable if they have identical relation patterns
                if np.array_equal(cs.order[i, :], cs.order[j, :]) and \
                   np.array_equal(cs.order[:, i], cs.order[:, j]):
                    ta += 1
        trans_autos.append(ta)
    print(f"    {label}: transposition-autos = {np.mean(trans_autos):.2f} ± {np.std(trans_autos):.2f}")

print(f"\n  Time: {time.time() - t0:.1f}s")
print("\n  VERDICT: Manifold-like causets should be 'rigid' — few automorphisms.")
print("  If 2-orders and sprinkled causets both have ~trivial Aut groups while")
print("  random DAGs don't, that's a geometric property.")


# ============================================================
# IDEA 206: ORDER POLYTOPE VOLUME (via linear extensions)
# ============================================================
print("\n" + "=" * 78)
print("IDEA 206: Order polytope volume ∝ 1/(# linear extensions)")
print("The order polytope P(C) has volume = (# linear extensions) / N!.")
print("More linear extensions = 'flatter' poset = higher volume.")
print("=" * 78)

def count_linear_extensions_small(order, N):
    """Count linear extensions by brute force for small N."""
    if N > 10:
        return -1  # Too expensive
    count = 0
    for perm in permutations(range(N)):
        valid = True
        for i in range(N):
            if not valid:
                break
            for j in range(i+1, N):
                # If perm[i] > perm[j] in the poset, but i appears before j in perm
                a, b = perm[i], perm[j]
                if order[b, a]:  # b < a in poset but a comes first
                    valid = False
                    break
        if valid:
            count += 1
    return count

def estimate_linear_extensions(order, N, n_samples=2000, rng=rng):
    """Estimate # linear extensions via random topological sort sampling."""
    # Karzanov-Khachiyan style: random topological sort, count how often it's valid
    # Actually just do repeated random topological sorts via Kahn's algorithm with random tie-breaking
    counts = Counter()
    for _ in range(n_samples):
        # Kahn's algorithm with random tie-breaking
        in_degree = np.sum(order, axis=0)
        available = list(np.where(in_degree == 0)[0])
        topo = []
        remaining = np.ones(N, dtype=bool)
        temp_in = in_degree.copy()
        while available:
            idx = rng.integers(len(available))
            node = available.pop(idx)
            topo.append(node)
            remaining[node] = False
            successors = np.where(order[node, :] & remaining)[0]
            for s in successors:
                temp_in[s] -= 1
                if temp_in[s] == 0:
                    available.append(s)
        counts[tuple(topo)] += 1

    # Number of distinct toposorts seen
    return len(counts)

t0 = time.time()

# Exact for small N
print("\n  Exact linear extension counts (small N):")
from math import factorial
for N_test in [5, 6, 7, 8]:
    n_trials = 30
    le_2order = []
    le_random = []
    le_chain = []
    le_antichain = []

    for _ in range(n_trials):
        _, cs = make_2order_causet(N_test, rng)
        le_2order.append(count_linear_extensions_small(cs.order, N_test))

        cs_r = random_poset(N_test, 0.25, rng)
        le_random.append(count_linear_extensions_small(cs_r.order, N_test))

    # Chain: only 1 linear extension
    # Antichain: N! linear extensions
    nfact = factorial(N_test)
    print(f"\n  N={N_test} (N!={nfact}):")
    print(f"    2-order:    LE = {np.mean(le_2order):.0f} ± {np.std(le_2order):.0f}  "
          f"(vol = {np.mean(le_2order)/nfact:.4f})")
    print(f"    random DAG: LE = {np.mean(le_random):.0f} ± {np.std(le_random):.0f}  "
          f"(vol = {np.mean(le_random)/nfact:.4f})")
    print(f"    (chain=1, antichain={nfact})")

print(f"\n  Time: {time.time() - t0:.1f}s")
print("\n  VERDICT: Linear extension count measures 'how constrained' the poset is.")
print("  Ordering fraction 1/3 for 2-orders should give a specific LE/N! ratio.")
print("  This connects to the entropy of the causal order.")


# ============================================================
# IDEA 207: CHAIN DECOMPOSITION STRUCTURE
# ============================================================
print("\n" + "=" * 78)
print("IDEA 207: Chain decomposition — Dilworth's min chain cover")
print("Partition the poset into minimum number of chains. Study chain length distribution.")
print("=" * 78)

def greedy_chain_decomposition(cs):
    """Greedy chain decomposition: repeatedly extract longest chain."""
    N = cs.n
    remaining = set(range(N))
    chains = []
    order = cs.order.copy()

    while remaining:
        # Find longest chain in remaining elements
        rem_list = sorted(remaining)
        dp = {x: 1 for x in rem_list}
        parent = {x: -1 for x in rem_list}

        for j in rem_list:
            for i in rem_list:
                if i >= j:
                    continue
                if order[i, j] and dp[i] + 1 > dp[j]:
                    dp[j] = dp[i] + 1
                    parent[j] = i

        # Extract longest chain
        best = max(rem_list, key=lambda x: dp[x])
        chain = []
        node = best
        while node != -1:
            chain.append(node)
            node = parent[node]
        chain.reverse()
        chains.append(chain)
        remaining -= set(chain)

    return chains

t0 = time.time()
N_test = 50
n_trials = 40

print(f"\n  Chain decomposition stats (N={N_test}, {n_trials} trials)")

for label, gen_func in [
    ("2-order", lambda: make_2order_causet(N_test, rng)[1]),
    ("random DAG (p=0.15)", lambda: random_poset(N_test, 0.15, rng)),
    ("sprinkled 2D", lambda: sprinkle_fast(N_test, dim=2, rng=rng)[0]),
]:
    n_chains_list = []
    max_chain_lens = []
    mean_chain_lens = []
    std_chain_lens = []

    for _ in range(n_trials):
        cs = gen_func()
        chains = greedy_chain_decomposition(cs)
        lengths = [len(c) for c in chains]
        n_chains_list.append(len(chains))
        max_chain_lens.append(max(lengths))
        mean_chain_lens.append(np.mean(lengths))
        std_chain_lens.append(np.std(lengths))

    print(f"\n  {label}:")
    print(f"    # chains:        {np.mean(n_chains_list):.1f} ± {np.std(n_chains_list):.1f}")
    print(f"    Longest chain:   {np.mean(max_chain_lens):.1f} ± {np.std(max_chain_lens):.1f}")
    print(f"    Mean chain len:  {np.mean(mean_chain_lens):.2f} ± {np.std(mean_chain_lens):.2f}")
    print(f"    Std chain len:   {np.mean(std_chain_lens):.2f} ± {np.std(std_chain_lens):.2f}")

print(f"\n  Time: {time.time() - t0:.1f}s")
print("\n  VERDICT: The chain decomposition captures how 'ordered' the poset is.")
print("  2-orders in 2D should have ~√N chains of length ~√N (balanced).")


# ============================================================
# IDEA 208: PATH ENUMERATION IN THE HASSE DIAGRAM
# ============================================================
print("\n" + "=" * 78)
print("IDEA 208: Path length distribution in the Hasse diagram (link graph)")
print("Count directed paths of each length. Path distribution = dimension estimator?")
print("=" * 78)

def path_length_distribution(cs, max_len=15):
    """Count directed paths of each length in the link graph."""
    links = cs.link_matrix()
    N = cs.n

    # Dynamic programming: paths[k][j] = number of paths of length k ending at j
    paths = np.zeros((max_len + 1, N), dtype=np.int64)
    paths[0, :] = 1  # paths of length 0: just the node itself

    counts = {0: N}
    link_int = links.astype(np.int64)

    for k in range(1, max_len + 1):
        # paths[k][j] = sum over i where link[i,j] of paths[k-1][i]
        paths[k, :] = link_int.T @ paths[k-1, :]
        total = int(np.sum(paths[k, :]))
        if total == 0:
            break
        counts[k] = total

    return counts

t0 = time.time()
N_test = 60
n_trials = 40

print(f"\n  Path length distribution (N={N_test}, {n_trials} trials)")

all_results = {}
for label, gen_func in [
    ("2-order", lambda: make_2order_causet(N_test, rng)[1]),
    ("random DAG (p=0.15)", lambda: random_poset(N_test, 0.15, rng)),
    ("sprinkled 2D", lambda: sprinkle_fast(N_test, dim=2, rng=rng)[0]),
    ("sprinkled 3D", lambda: sprinkle_fast(N_test, dim=3, rng=rng)[0]),
]:
    all_dists = defaultdict(list)
    for _ in range(n_trials):
        cs = gen_func()
        dist = path_length_distribution(cs, max_len=20)
        for k, v in dist.items():
            all_dists[k].append(v)

    print(f"\n  {label}:")
    for k in sorted(all_dists.keys()):
        if k == 0:
            continue
        vals = all_dists[k]
        print(f"    len {k:2d}: {np.mean(vals):10.1f} ± {np.std(vals):8.1f}")
        if np.mean(vals) < 1:
            break

    # Compute "path entropy" — entropy of path length distribution
    entropies = []
    for trial in range(n_trials):
        dist_vec = []
        for k in sorted(all_dists.keys()):
            if k == 0:
                continue
            if trial < len(all_dists[k]):
                dist_vec.append(all_dists[k][trial])
        dist_vec = np.array(dist_vec, dtype=float)
        if np.sum(dist_vec) > 0:
            p = dist_vec / np.sum(dist_vec)
            p = p[p > 0]
            entropies.append(-np.sum(p * np.log(p)))

    print(f"    Path entropy: {np.mean(entropies):.3f} ± {np.std(entropies):.3f}")
    all_results[label] = np.mean(entropies)

print(f"\n  Time: {time.time() - t0:.1f}s")
print("\n  VERDICT: Path entropy measures the 'spread' of geodesic lengths.")
print("  Higher-d causets should have shorter paths (more links per element).")
print("  This is a clean, computable dimension estimator from pure geometry.")


# ============================================================
# IDEA 209: MÖBIUS FUNCTION STATISTICS
# ============================================================
print("\n" + "=" * 78)
print("IDEA 209: Möbius function μ(x,y) statistics")
print("The Möbius function of the poset encodes inclusion-exclusion structure.")
print("For intervals: μ(x,y) = -Σ_{x≤z<y} μ(x,z), μ(x,x)=1.")
print("=" * 78)

def mobius_function(order, N):
    """Compute Möbius function μ(x,y) for all pairs."""
    mu = np.zeros((N, N), dtype=np.int64)
    for x in range(N):
        mu[x, x] = 1
        for y in range(x+1, N):
            if not order[x, y]:
                continue
            # μ(x,y) = -Σ_{z: x≤z<y, z≠y} μ(x,z)
            val = 0
            for z in range(x, y):
                if z == x or order[x, z]:
                    if order[z, y] or z == y:
                        # z is between x and y (or z == x)
                        val -= mu[x, z]
            mu[x, y] = val
    return mu

t0 = time.time()
N_test = 30
n_trials = 40

print(f"\n  Möbius function statistics (N={N_test}, {n_trials} trials)")

for label, gen_func in [
    ("2-order", lambda: make_2order_causet(N_test, rng)[1]),
    ("random DAG (p=0.15)", lambda: random_poset(N_test, 0.15, rng)),
    ("sprinkled 2D", lambda: sprinkle_fast(N_test, dim=2, rng=rng)[0]),
]:
    means, stds, max_abs, frac_nonzero = [], [], [], []
    frac_minus1, frac_plus1 = [], []

    for _ in range(n_trials):
        cs = gen_func()
        mu = mobius_function(cs.order, cs.n)
        # Get values for related pairs only
        related = np.where(np.triu(cs.order, k=1))
        if len(related[0]) == 0:
            continue
        vals = mu[related]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
        max_abs.append(np.max(np.abs(vals)))
        frac_nonzero.append(np.mean(vals != 0))
        frac_minus1.append(np.mean(vals == -1))
        frac_plus1.append(np.mean(vals == 1))

    print(f"\n  {label}:")
    print(f"    Mean μ:       {np.mean(means):.4f} ± {np.std(means):.4f}")
    print(f"    Std μ:        {np.mean(stds):.4f} ± {np.std(stds):.4f}")
    print(f"    Max |μ|:      {np.mean(max_abs):.1f} ± {np.std(max_abs):.1f}")
    print(f"    Frac nonzero: {np.mean(frac_nonzero):.4f}")
    print(f"    Frac μ=-1:    {np.mean(frac_minus1):.4f}")
    print(f"    Frac μ=+1:    {np.mean(frac_plus1):.4f}")

print(f"\n  Time: {time.time() - t0:.1f}s")
print("\n  VERDICT: The Möbius function encodes fine-grained incidence structure.")
print("  For a chain, μ(x,y)=(-1)^{y-x}. For richer posets, the distribution of μ")
print("  values should distinguish manifold-like from random. If max|μ| grows")
print("  differently with N, that's a structural theorem.")


# ============================================================
# IDEA 210: CONVEX SUBPOSET DENSITY
# ============================================================
print("\n" + "=" * 78)
print("IDEA 210: Convex subposet density")
print("A subset S is convex if x,y∈S and x<z<y implies z∈S.")
print("What fraction of k-subsets are convex? Measures geodesic completeness.")
print("=" * 78)

def is_convex_subset(order, N, subset):
    """Check if subset is convex in the poset."""
    subset_set = set(subset)
    for x in subset:
        for y in subset:
            if not order[x, y]:
                continue
            # Check all z between x and y
            for z in range(N):
                if z in subset_set:
                    continue
                if order[x, z] and order[z, y]:
                    return False
    return True

def convex_density(cs, k, n_samples=500, rng=rng):
    """Estimate fraction of k-subsets that are convex."""
    N = cs.n
    if N < k:
        return 0.0
    convex_count = 0
    elements = list(range(N))
    for _ in range(n_samples):
        subset = sorted(rng.choice(N, size=k, replace=False))
        if is_convex_subset(cs.order, N, subset):
            convex_count += 1
    return convex_count / n_samples

t0 = time.time()
N_test = 35
n_trials = 30

print(f"\n  Convex subset density (N={N_test}, {n_trials} trials)")

for k in [3, 4, 5]:
    print(f"\n  k = {k}:")
    for label, gen_func in [
        ("2-order", lambda: make_2order_causet(N_test, rng)[1]),
        ("random DAG (p=0.15)", lambda: random_poset(N_test, 0.15, rng)),
        ("sprinkled 2D", lambda: sprinkle_fast(N_test, dim=2, rng=rng)[0]),
    ]:
        densities = []
        for _ in range(n_trials):
            cs = gen_func()
            d = convex_density(cs, k, n_samples=300, rng=rng)
            densities.append(d)
        print(f"    {label}: convex fraction = {np.mean(densities):.4f} ± {np.std(densities):.4f}")

print(f"\n  Time: {time.time() - t0:.1f}s")
print("\n  VERDICT: Convex density measures how 'interval-like' the poset's substructure is.")
print("  Manifold-like causets should have higher convex density (geodesic completeness).")
print("  Random DAGs should have lower convexity (arbitrary missing intermediate elements).")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 78)
print("ROUND 11 SUMMARY: Ideas 201-210")
print("=" * 78)
print("""
All 10 ideas are PURE CAUSAL GEOMETRY — no SJ vacuum used.

201. Pattern densities (2-chain, V, Λ, diamond): quantifies local poset structure
202. Width profile: captures bulk geometry shape
203. Ramsey-type bounds: chain × antichain product scaling
204. Forbidden subposets (N, Z): structural fingerprints of 2D embeddability
205. Automorphism group: rigidity of manifold-like causets
206. Order polytope volume: entropy of causal order via linear extensions
207. Chain decomposition: balanced decomposition structure
208. Path enumeration: dimension estimator from Hasse diagram paths
209. Möbius function: fine-grained incidence structure
210. Convex subposet density: geodesic completeness measure
""")

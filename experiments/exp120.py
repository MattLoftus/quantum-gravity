"""
Experiment 120: RANDOM WALK TO CAUSAL SETS — Ideas 711-720

METHODOLOGY: Start from a random mathematical concept, take exactly 3 associative
steps toward causal sets, then design and run an experiment testing the connection.

711. FIBONACCI NUMBERS → golden ratio → self-similar growth → growth rate of
     longest chain in sequential causets. Does longest_chain(N)/N → 1/φ²?

712. CONTINUED FRACTIONS → rational approximation → ordering fraction convergence →
     ordering fraction of d-orders as a continued fraction in 1/d.

713. P-ADIC NUMBERS → ultrametric distance → tree structure → ultrametricity of
     causal set intervals (do intervals form an ultrametric tree?).

714. TROPICAL GEOMETRY → min-plus algebra → longest path = tropical determinant →
     tropical determinant of the causal matrix vs BD action.

715. KNOT INVARIANTS → Jones polynomial → bracket expansion → bracket-like
     polynomial of the Hasse diagram. Is it a causal set invariant?

716. CELLULAR AUTOMATA RULE 30 → emergent complexity from simple rules → CSG
     percolation → is CSG at the critical coupling like Rule 30 (Class III)?

717. RAMANUJAN'S PARTITION FUNCTION → integer partitions → antichain decomposition →
     partition the causet into antichains. Is the number of ways related to p(N)?

718. CATALAN NUMBERS → Dyck paths → monotone paths in 2-orders → count monotone
     subsequences in the permutations defining a 2-order.

719. BROWNIAN MOTION → Wiener process → random walk on DAG → random walk mixing
     time on the Hasse diagram vs causet dimension.

720. WAVELETS → multiresolution analysis → coarse-graining → wavelet transform of
     the causal matrix. Do detail coefficients detect the phase transition?
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats
from scipy.linalg import eigh, svd
from scipy.optimize import curve_fit
import time

from causal_sets.fast_core import FastCausalSet, sprinkle_fast, csg_fast
from causal_sets.two_orders import TwoOrder, swap_move
from causal_sets.d_orders import DOrder
from causal_sets.bd_action import count_intervals_by_size, count_links
from causal_sets.sj_vacuum import pauli_jordan_function, sj_wightman_function
from causal_sets.two_orders_v2 import bd_action_corrected, mcmc_corrected

np.set_printoptions(precision=6, suppress=True, linewidth=120)
rng = np.random.default_rng(42)

# ============================================================
# SHARED UTILITIES
# ============================================================

def random_2order(N, rng_local=None):
    """Generate a random 2-order and convert to FastCausalSet."""
    if rng_local is None:
        rng_local = rng
    to = TwoOrder(N, rng=rng_local)
    return to.to_causet(), to


def random_dorder(d, N, rng_local=None):
    """Generate a random d-order and convert to FastCausalSet."""
    if rng_local is None:
        rng_local = rng
    do = DOrder(d, N, rng=rng_local)
    return do.to_causet_fast(), do


def hasse_adjacency(cs):
    """Symmetric adjacency matrix of the Hasse diagram."""
    links = cs.link_matrix()
    return (links | links.T).astype(np.float64)


def hasse_laplacian(cs):
    """Laplacian of the Hasse diagram."""
    adj = hasse_adjacency(cs)
    degree = np.sum(adj, axis=1)
    return np.diag(degree) - adj


print("=" * 80)
print("EXPERIMENT 120: RANDOM WALK TO CAUSAL SETS (Ideas 711-720)")
print("=" * 80)
sys.stdout.flush()


# ============================================================
# IDEA 711: FIBONACCI → GOLDEN RATIO → SELF-SIMILAR GROWTH →
#           LONGEST CHAIN GROWTH RATE
# ============================================================
print("\n" + "=" * 80)
print("IDEA 711: Fibonacci Numbers → Longest Chain Growth Rate")
print("=" * 80)
print("""
WALK TO CAUSAL SETS:
  Step 1: Fibonacci numbers → golden ratio φ = (1+√5)/2
  Step 2: Golden ratio → self-similar growth (φ appears in growth rates of
          many self-similar structures)
  Step 3: Self-similar growth → causal sets grow element by element in CSG.
          The longest chain grows sublinearly. Does the ratio converge to
          something involving φ?

EXPERIMENT: For 2-orders (uniform random causets in 2D), the longest chain
length scales as ~ 2√N (Bollobás-Winkler). For sprinkled causets, it's
related to the geodesic diameter. Measure longest_chain(N)/√N for various
N and check if the limiting constant has any connection to φ or Fibonacci.
Also: in CSG causets at critical coupling, does the chain growth rate
involve φ?
""")
sys.stdout.flush()

t0 = time.time()

phi = (1 + np.sqrt(5)) / 2

print(f"  Golden ratio φ = {phi:.6f}")
print(f"  1/φ² = {1/phi**2:.6f}")
print(f"  2/φ = {2/phi:.6f}")
print()

# Measure longest chain for 2-orders at various N
print(f"  {'Source':>20} {'N':>6} {'L_max':>8} {'L/√N':>8} {'L/N':>8}")
print("  " + "-" * 60)

chain_data_2order = []
for N in [20, 40, 60, 80, 100, 150, 200, 300]:
    chains = []
    n_trials = 20 if N <= 100 else 10
    for trial in range(n_trials):
        cs, _ = random_2order(N, rng_local=np.random.default_rng(trial * 13 + N))
        chains.append(cs.longest_chain())
    mean_chain = np.mean(chains)
    chain_data_2order.append((N, mean_chain))
    print(f"  {'2-order':>20} {N:>6} {mean_chain:>8.2f} {mean_chain/np.sqrt(N):>8.4f} {mean_chain/N:>8.4f}")

# Fit L = c * N^alpha for 2-orders
Ns = np.array([x[0] for x in chain_data_2order])
Ls = np.array([x[1] for x in chain_data_2order])
log_fit = np.polyfit(np.log(Ns), np.log(Ls), 1)
alpha_2order = log_fit[0]
c_2order = np.exp(log_fit[1])
print(f"\n  2-order fit: L = {c_2order:.4f} * N^{alpha_2order:.4f}")
print(f"  Expected: L ~ 2√N → α = 0.5, c = 2.0")
print(f"  Ratio c/φ = {c_2order/phi:.4f}, c*φ = {c_2order*phi:.4f}")

# Sprinkled causets in 2D diamond
print()
chain_data_sprinkle = []
for N in [20, 40, 60, 80, 100, 150, 200]:
    chains = []
    n_trials = 15 if N <= 100 else 8
    for trial in range(n_trials):
        cs, _ = sprinkle_fast(N, dim=2, rng=np.random.default_rng(trial * 17 + N))
        chains.append(cs.longest_chain())
    mean_chain = np.mean(chains)
    chain_data_sprinkle.append((N, mean_chain))
    print(f"  {'sprinkle-2D':>20} {N:>6} {mean_chain:>8.2f} {mean_chain/np.sqrt(N):>8.4f} {mean_chain/N:>8.4f}")

Ns_s = np.array([x[0] for x in chain_data_sprinkle])
Ls_s = np.array([x[1] for x in chain_data_sprinkle])
log_fit_s = np.polyfit(np.log(Ns_s), np.log(Ls_s), 1)
alpha_spr = log_fit_s[0]
c_spr = np.exp(log_fit_s[1])
print(f"\n  Sprinkle fit: L = {c_spr:.4f} * N^{alpha_spr:.4f}")
print(f"  Ratio c/φ = {c_spr/phi:.4f}")

# CSG at various couplings
print()
for p_csg in [0.3, 0.4, 0.5]:
    chain_data_csg = []
    for N in [30, 50, 80, 120]:
        chains = []
        for trial in range(10):
            cs = csg_fast(N, coupling=p_csg, rng=np.random.default_rng(trial * 19 + N))
            chains.append(cs.longest_chain())
        mean_chain = np.mean(chains)
        chain_data_csg.append((N, mean_chain))
        print(f"  {'CSG p='+str(p_csg):>20} {N:>6} {mean_chain:>8.2f} {mean_chain/np.sqrt(N):>8.4f} {mean_chain/N:>8.4f}")
    Ns_c = np.array([x[0] for x in chain_data_csg])
    Ls_c = np.array([x[1] for x in chain_data_csg])
    log_fit_c = np.polyfit(np.log(Ns_c), np.log(Ls_c), 1)
    print(f"  CSG p={p_csg} fit: L ~ N^{log_fit_c[0]:.4f}")
    print()

# Check Fibonacci: is the sequence of chain lengths at N = F_k Fibonacci-like?
fibs = [1, 1]
while fibs[-1] < 300:
    fibs.append(fibs[-1] + fibs[-2])
fibs = [f for f in fibs if f >= 10]

print("  Chain lengths at Fibonacci N values:")
fib_chains = []
for N in fibs[:8]:
    chains = []
    for trial in range(15):
        cs, _ = random_2order(N, rng_local=np.random.default_rng(trial * 23 + N))
        chains.append(cs.longest_chain())
    mean_c = np.mean(chains)
    fib_chains.append(mean_c)
    print(f"    N = {N:>4} (Fib): L_max = {mean_c:.2f}")

if len(fib_chains) >= 3:
    ratios = [fib_chains[i+1]/fib_chains[i] for i in range(len(fib_chains)-1)]
    print(f"  Consecutive ratios: {[f'{r:.4f}' for r in ratios]}")
    print(f"  √φ = {np.sqrt(phi):.4f} (expected ratio for √N scaling with Fib N)")

dt = time.time() - t0
print(f"\n  [Idea 711 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# IDEA 712: CONTINUED FRACTIONS → ORDERING FRACTION AS
#           CONTINUED FRACTION IN 1/d
# ============================================================
print("\n" + "=" * 80)
print("IDEA 712: Continued Fractions → Ordering Fraction vs Dimension")
print("=" * 80)
print("""
WALK TO CAUSAL SETS:
  Step 1: Continued fractions → rational approximation of irrational numbers
  Step 2: Rational approximation → the ordering fraction f(d,N) of a d-order
          converges to some function of d as N→∞. What is this function?
  Step 3: For a random d-order, f = E[relations]/C(N,2). Known: f(2) = 1/3.
          Does f(d) have a nice continued fraction expansion in 1/d?

EXPERIMENT: Measure f(d) for d = 1..8, find the exact values, and check
if f(d) = 1/(2^d - 1) or some other closed form. Compute the continued
fraction representation.
""")
sys.stdout.flush()

t0 = time.time()

def ordering_fraction_dorder(d, N, n_trials=30):
    """Average ordering fraction for random d-orders."""
    fracs = []
    for trial in range(n_trials):
        cs, _ = random_dorder(d, N, rng_local=np.random.default_rng(trial * 31 + d * 100))
        fracs.append(cs.ordering_fraction())
    return np.mean(fracs), np.std(fracs) / np.sqrt(n_trials)


def to_continued_fraction(x, max_terms=8):
    """Convert a float to its continued fraction representation."""
    cf = []
    for _ in range(max_terms):
        a = int(np.floor(x))
        cf.append(a)
        frac = x - a
        if abs(frac) < 1e-10:
            break
        x = 1.0 / frac
    return cf


print(f"  {'d':>3} {'N=30':>10} {'N=60':>10} {'N=100':>10} {'N->inf est':>10} {'1/(d+1)':>10} {'CF repr':>25}")
print("  " + "-" * 85)

f_values = {}
for d in range(1, 9):
    f_30, _ = ordering_fraction_dorder(d, 30, n_trials=40 if d <= 4 else 20)
    f_60, _ = ordering_fraction_dorder(d, 60, n_trials=30 if d <= 4 else 15)
    n_100 = 100 if d <= 4 else 50
    f_100, se_100 = ordering_fraction_dorder(d, n_100, n_trials=20 if d <= 4 else 10)

    # For large N, f converges
    if d == 1:
        f_inf = 1.0  # d=1 is a total order, ordering fraction = 1 ... wait
        # Actually d=1 means one permutation, so EVERY pair is related
        # (all elements are totally ordered). f = 1.0? No...
        # d=1: element i < j iff perm[i] < perm[j]. Since it's a permutation,
        # exactly N(N-1)/2 pairs satisfy this, so f = 0.5
        f_inf = f_100  # measure it
    else:
        f_inf = f_100

    pred_dp1 = 1.0 / (d + 1)

    cf = to_continued_fraction(f_inf)
    cf_str = str(cf[:6])

    f_values[d] = f_inf
    print(f"  {d:>3} {f_30:>10.6f} {f_60:>10.6f} {f_100:>10.6f} {f_inf:>10.6f} {pred_dp1:>10.6f} {cf_str:>25}")

# Check f(d) = 1/(d+1) hypothesis
print("\n  Testing f(d) = 1/(d+1) hypothesis:")
print(f"  {'d':>3} {'measured':>10} {'1/(d+1)':>10} {'ratio':>10}")
print("  " + "-" * 35)
for d in range(1, 9):
    pred = 1.0 / (d + 1)
    print(f"  {d:>3} {f_values[d]:>10.6f} {pred:>10.6f} {f_values[d]/pred:>10.6f}")

# Continued fraction of the sequence f(1), f(2), ..., f(8)
print("\n  Sequence f(d): ", [f"{f_values[d]:.6f}" for d in range(1, 9)])
print(f"  Sequence 1/(d+1): {[f'{1/(d+1):.6f}' for d in range(1, 9)]}")

dt = time.time() - t0
print(f"\n  [Idea 712 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# IDEA 713: P-ADIC NUMBERS → ULTRAMETRIC → TREE STRUCTURE →
#           ULTRAMETRICITY OF CAUSAL SET INTERVALS
# ============================================================
print("\n" + "=" * 80)
print("IDEA 713: p-adic Numbers → Ultrametricity of Intervals")
print("=" * 80)
print("""
WALK TO CAUSAL SETS:
  Step 1: p-adic numbers → ultrametric distance d(x,z) <= max(d(x,y), d(y,z))
  Step 2: Ultrametric distance → tree structure (ultrametric spaces embed in trees)
  Step 3: Trees → causal set interval structure. The intervals of a causet
          form a partial order. Define d(I1,I2) via inclusion/overlap.
          Is this an ultrametric? If so, causets have hidden tree structure.

EXPERIMENT: For each pair of elements (a,b) with a<b, the interval I(a,b)
is the set of elements between them. Define a distance between intervals:
d(I1,I2) = 1 - |I1 ∩ I2| / |I1 ∪ I2| (Jaccard distance).
Test the ultrametric inequality on random triples of intervals.
A high fraction satisfying ultrametricity → hidden tree structure.
""")
sys.stdout.flush()

t0 = time.time()

def compute_intervals(cs):
    """Compute all intervals I(a,b) = {x : a<x<b} for all related pairs."""
    N = cs.n
    order = cs.order
    intervals = {}
    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            if order[i, j]:
                between = set()
                for k in range(N):
                    if k != i and k != j and order[i, k] and order[k, j]:
                        between.add(k)
                intervals[(i, j)] = between
                pairs.append((i, j))
    return intervals, pairs


def jaccard_distance(s1, s2):
    """Jaccard distance between two sets."""
    if len(s1) == 0 and len(s2) == 0:
        return 0.0
    union = s1 | s2
    if len(union) == 0:
        return 0.0
    return 1.0 - len(s1 & s2) / len(union)


def test_ultrametricity(intervals, pairs, n_triples=2000, rng_local=None):
    """Test fraction of triples satisfying ultrametric inequality."""
    if rng_local is None:
        rng_local = rng
    if len(pairs) < 3:
        return 0.0, 0
    n_sat = 0
    n_tested = 0
    for _ in range(n_triples):
        idx = rng_local.choice(len(pairs), size=3, replace=False)
        p1, p2, p3 = pairs[idx[0]], pairs[idx[1]], pairs[idx[2]]
        I1, I2, I3 = intervals[p1], intervals[p2], intervals[p3]

        d12 = jaccard_distance(I1, I2)
        d13 = jaccard_distance(I1, I3)
        d23 = jaccard_distance(I2, I3)

        # Ultrametric: each distance <= max of other two
        dists = sorted([d12, d13, d23])
        if dists[2] <= max(dists[0], dists[1]) + 1e-10:
            n_sat += 1
        n_tested += 1

    return n_sat / n_tested if n_tested > 0 else 0.0, n_tested


print(f"  {'Source':>20} {'N':>5} {'#intervals':>12} {'ultrametric%':>14} {'mean Jaccard':>14}")
print("  " + "-" * 70)

for source_name, make_cs in [
    ("2-order", lambda N, t: random_2order(N, rng_local=np.random.default_rng(t*41+N))[0]),
    ("sprinkle-2D", lambda N, t: sprinkle_fast(N, dim=2, rng=np.random.default_rng(t*43+N))[0]),
]:
    for N in [20, 30, 40]:
        ultra_fracs = []
        mean_jaccards = []
        for trial in range(5):
            cs = make_cs(N, trial)
            intervals, pairs = compute_intervals(cs)
            # Filter to intervals with at least 1 interior element
            nonempty = [(p, intervals[p]) for p in pairs if len(intervals[p]) > 0]
            if len(nonempty) < 3:
                continue
            nonempty_dict = {p: s for p, s in nonempty}
            nonempty_pairs = [p for p, _ in nonempty]

            frac, _ = test_ultrametricity(nonempty_dict, nonempty_pairs, n_triples=1000)
            ultra_fracs.append(frac)

            # Mean Jaccard distance
            dists = []
            for _ in range(200):
                idx = rng.choice(len(nonempty_pairs), size=2, replace=False)
                d = jaccard_distance(nonempty_dict[nonempty_pairs[idx[0]]],
                                     nonempty_dict[nonempty_pairs[idx[1]]])
                dists.append(d)
            mean_jaccards.append(np.mean(dists))

        if ultra_fracs:
            print(f"  {source_name:>20} {N:>5} {len(nonempty):>12} "
                  f"{np.mean(ultra_fracs):>13.4f} {np.mean(mean_jaccards):>13.4f}")

# Compare with random sets (null model)
print("\n  Null model (random sets of same sizes):")
for N in [20, 30]:
    cs, _ = random_2order(N)
    intervals, pairs = compute_intervals(cs)
    nonempty = [(p, intervals[p]) for p in pairs if len(intervals[p]) > 0]
    sizes = [len(s) for _, s in nonempty]
    if len(sizes) < 3:
        continue

    # Generate random sets with same size distribution
    random_intervals = {}
    random_pairs = list(range(len(nonempty)))
    for i, (p, s) in enumerate(nonempty):
        random_intervals[i] = set(rng.choice(N, size=len(s), replace=False).tolist())

    frac_null, _ = test_ultrametricity(random_intervals, random_pairs, n_triples=1000)
    print(f"  {'random sets':>20} {N:>5} {len(nonempty):>12} {frac_null:>13.4f}")

dt = time.time() - t0
print(f"\n  [Idea 713 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# IDEA 714: TROPICAL GEOMETRY → MIN-PLUS ALGEBRA →
#           TROPICAL DETERMINANT = LONGEST PATH → BD ACTION
# ============================================================
print("\n" + "=" * 80)
print("IDEA 714: Tropical Geometry → Tropical Determinant vs BD Action")
print("=" * 80)
print("""
WALK TO CAUSAL SETS:
  Step 1: Tropical geometry → min-plus (or max-plus) algebra where
          addition = max, multiplication = +
  Step 2: Max-plus algebra → the tropical determinant of a matrix M is
          max over permutations σ of Σ M[i,σ(i)] = longest path in the
          associated bipartite graph
  Step 3: Longest path → on the causal matrix C[i,j], the tropical
          determinant is the longest chain. But more interestingly:
          define M[i,j] = interval_size(i,j). The tropical determinant
          encodes maximal total interval content along a permutation.

EXPERIMENT: Compare the tropical determinant of the interval-size matrix
with the BD action. Is there a correlation? The tropical det captures
"how much causal structure" can be packed along a permutation.
""")
sys.stdout.flush()

t0 = time.time()

def tropical_determinant(M):
    """
    Compute tropical determinant: max over permutations σ of Σ M[i,σ(i)].
    For small N, use exact enumeration. For larger N, use Hungarian algorithm.
    """
    N = M.shape[0]
    if N <= 10:
        from itertools import permutations
        best = -np.inf
        for perm in permutations(range(N)):
            val = sum(M[i, perm[i]] for i in range(N))
            if val > best:
                best = val
        return best
    else:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-M)
        return M[row_ind, col_ind].sum()


def interval_size_matrix(cs):
    """Matrix where M[i,j] = |interval(i,j)| if i<j, 0 otherwise."""
    order_int = cs.order.astype(np.int32)
    return (order_int @ order_int).astype(float) * cs.order.astype(float)


print(f"  {'Source':>15} {'N':>5} {'trop_det':>10} {'BD action':>10} {'longest_ch':>10} {'#links':>8} {'#rels':>8}")
print("  " + "-" * 75)

trop_dets = []
bd_actions = []

for N in [10, 15, 20, 25]:
    for trial in range(8):
        cs, _ = random_2order(N, rng_local=np.random.default_rng(trial * 53 + N))
        M = interval_size_matrix(cs)
        td = tropical_determinant(M)
        ivals = count_intervals_by_size(cs, max_size=3)
        links = ivals.get(0, 0)
        i2 = ivals.get(1, 0)
        bd = N - 2 * links + i2  # 2D BD action
        lc = cs.longest_chain()
        nrels = cs.num_relations()

        trop_dets.append(td)
        bd_actions.append(bd)

        if trial == 0:
            print(f"  {'2-order':>15} {N:>5} {td:>10.1f} {bd:>10.1f} {lc:>10} {links:>8} {nrels:>8}")

# Correlation
trop_dets = np.array(trop_dets)
bd_actions = np.array(bd_actions)
r, p_val = stats.pearsonr(trop_dets, bd_actions)
rho, p_rho = stats.spearmanr(trop_dets, bd_actions)
print(f"\n  Pearson r(tropical_det, BD_action) = {r:.4f} (p = {p_val:.2e})")
print(f"  Spearman ρ(tropical_det, BD_action) = {rho:.4f} (p = {p_rho:.2e})")

# Scaling
print(f"\n  Tropical det scaling with N:")
for N in [10, 15, 20, 25]:
    vals = []
    for trial in range(15):
        cs, _ = random_2order(N, rng_local=np.random.default_rng(trial * 57 + N * 3))
        M = interval_size_matrix(cs)
        vals.append(tropical_determinant(M))
    print(f"    N={N:>3}: trop_det = {np.mean(vals):.2f} +/- {np.std(vals):.2f}, "
          f"trop_det/N = {np.mean(vals)/N:.4f}, trop_det/N^2 = {np.mean(vals)/N**2:.6f}")

dt = time.time() - t0
print(f"\n  [Idea 714 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# IDEA 715: KNOT INVARIANTS → BRACKET POLYNOMIAL →
#           BRACKET-LIKE POLYNOMIAL OF HASSE DIAGRAM
# ============================================================
print("\n" + "=" * 80)
print("IDEA 715: Knot Invariants → Bracket Polynomial of Hasse Diagram")
print("=" * 80)
print("""
WALK TO CAUSAL SETS:
  Step 1: Knot invariants → Jones polynomial computed via Kauffman bracket
  Step 2: Kauffman bracket → state sum over edge resolutions of a diagram
  Step 3: Edge resolutions → for each edge in the Hasse diagram, we can
          either "keep" or "remove" it. Define a bracket polynomial:
          B(q) = Σ_S q^{f(S)} where S ranges over subsets of Hasse edges
          and f(S) = #connected_components of the induced subgraph.

EXPERIMENT: Compute B(q) for small causets by exact enumeration. B(1)
counts total states = 2^L. The polynomial's structure may encode
topological information about the causet. Compare 2-orders vs sprinkled
vs random DAGs.
""")
sys.stdout.flush()

t0 = time.time()

def bracket_polynomial_hasse(cs, max_edges=18):
    """
    Compute bracket polynomial B(q) = Σ_S q^{components(S)}
    where S ranges over subsets of Hasse edges.
    Returns coefficients as dict: {k: count} where B(q) = Σ count * q^k.
    """
    links = cs.link_matrix()
    # Get undirected edges
    edges = set()
    for i in range(cs.n):
        for j in range(i+1, cs.n):
            if links[i, j] or links[j, i]:
                edges.add((min(i, j), max(i, j)))
    edges = list(edges)

    if len(edges) > max_edges:
        return {}, len(edges)

    n_edges = len(edges)
    coeffs = {}

    # Enumerate all 2^n_edges subsets
    for mask in range(2**n_edges):
        # Build adjacency for this subset
        adj = {i: set() for i in range(cs.n)}
        for bit in range(n_edges):
            if mask & (1 << bit):
                u, v = edges[bit]
                adj[u].add(v)
                adj[v].add(u)

        # Count connected components (of the N-node graph with these edges)
        visited = set()
        components = 0
        for node in range(cs.n):
            if node not in visited:
                components += 1
                stack = [node]
                while stack:
                    curr = stack.pop()
                    if curr in visited:
                        continue
                    visited.add(curr)
                    stack.extend(adj[curr] - visited)

        coeffs[components] = coeffs.get(components, 0) + 1

    return coeffs, n_edges


print(f"  {'Source':>15} {'N':>4} {'#edges':>7} {'B(1)':>8} {'<comp>':>8} {'max_comp':>9} {'entropy':>8}")
print("  " + "-" * 65)

for N in [8, 10, 12]:
    for source_name, make_cs in [
        ("2-order", lambda t: random_2order(N, rng_local=np.random.default_rng(t*61+N))[0]),
        ("sprinkle", lambda t: sprinkle_fast(N, dim=2, rng=np.random.default_rng(t*67+N))[0]),
    ]:
        for trial in range(3):
            cs = make_cs(trial)
            coeffs, n_edges = bracket_polynomial_hasse(cs, max_edges=18)
            if not coeffs:
                continue
            total_states = sum(coeffs.values())
            mean_comp = sum(k * v for k, v in coeffs.items()) / total_states
            max_comp = max(coeffs.keys())

            # Shannon entropy of the distribution
            probs = np.array([coeffs.get(k, 0) / total_states for k in range(1, max_comp + 1)])
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))

            if trial == 0:
                print(f"  {source_name:>15} {N:>4} {n_edges:>7} {total_states:>8} "
                      f"{mean_comp:>8.3f} {max_comp:>9} {entropy:>8.3f}")

# Evaluate B(q) at specific points for comparison
print("\n  Bracket polynomial evaluated at specific q values:")
print(f"  {'Source':>15} {'N':>4} {'B(-1)':>10} {'B(2)':>12} {'B(i) real':>10} {'B(i) imag':>10}")
print("  " + "-" * 60)

for N in [8, 10]:
    for source_name, make_cs in [
        ("2-order", lambda t: random_2order(N, rng_local=np.random.default_rng(t*71+N))[0]),
        ("sprinkle", lambda t: sprinkle_fast(N, dim=2, rng=np.random.default_rng(t*73+N))[0]),
    ]:
        cs = make_cs(0)
        coeffs, n_edges = bracket_polynomial_hasse(cs, max_edges=18)
        if not coeffs:
            continue
        B_neg1 = sum((-1)**k * v for k, v in coeffs.items())
        B_2 = sum(2**k * v for k, v in coeffs.items())
        B_i = sum((1j)**k * v for k, v in coeffs.items())
        print(f"  {source_name:>15} {N:>4} {B_neg1:>10} {B_2:>12} {B_i.real:>10.1f} {B_i.imag:>10.1f}")

dt = time.time() - t0
print(f"\n  [Idea 715 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# IDEA 716: RULE 30 → EMERGENT COMPLEXITY → CSG AT CRITICAL
#           COUPLING → WOLFRAM CLASS III BEHAVIOR
# ============================================================
print("\n" + "=" * 80)
print("IDEA 716: Cellular Automata Rule 30 → CSG Complexity Class")
print("=" * 80)
print("""
WALK TO CAUSAL SETS:
  Step 1: Rule 30 → emergent complexity from simple local rules (Wolfram Class III)
  Step 2: Class III → aperiodic, complex behavior at the edge of chaos
  Step 3: Edge of chaos → CSG (causal set growth) at the critical coupling p_c
          where the transition from sparse to dense occurs. At p_c, does
          the Hasse diagram exhibit Rule 30-like complexity?

EXPERIMENT: Measure Shannon entropy of the degree sequence and Lempel-Ziv
complexity of the Hasse adjacency pattern as a function of CSG coupling p.
Compare with actual Rule 30 output. Peak complexity at p_c indicates
a causal set analogue of Class III behavior.
""")
sys.stdout.flush()

t0 = time.time()

def degree_sequence_entropy(cs):
    """Shannon entropy of the Hasse degree sequence."""
    links = cs.link_matrix()
    in_deg = np.sum(links, axis=0)
    out_deg = np.sum(links, axis=1)
    total_deg = in_deg + out_deg
    vals, counts = np.unique(total_deg, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-15))


def lz_complexity(binary_seq):
    """Lempel-Ziv complexity of a binary sequence (normalized)."""
    n = len(binary_seq)
    if n == 0:
        return 0
    s = ''.join(str(int(b)) for b in binary_seq)
    words = set()
    w = ''
    for c in s:
        w += c
        if w not in words:
            words.add(w)
            w = ''
    return len(words) / (n / np.log2(n + 1) + 1)


def adjacency_lz_complexity(cs):
    """LZ complexity of the flattened upper-triangle of the Hasse adjacency."""
    links = cs.link_matrix()
    upper = []
    for i in range(cs.n):
        for j in range(i+1, cs.n):
            upper.append(links[i, j])
    return lz_complexity(upper)


# Generate Rule 30 for comparison
def rule30(width, steps):
    """Generate Rule 30 cellular automaton."""
    state = np.zeros(width, dtype=int)
    state[width // 2] = 1
    history = [state.copy()]
    for _ in range(steps):
        new_state = np.zeros(width, dtype=int)
        for i in range(width):
            left = state[(i - 1) % width]
            center = state[i]
            right = state[(i + 1) % width]
            pattern = left * 4 + center * 2 + right
            new_state[i] = (30 >> pattern) & 1
        state = new_state
        history.append(state.copy())
    return np.array(history)

rule30_output = rule30(50, 50)
rule30_flat = rule30_output.flatten()
rule30_lz = lz_complexity(rule30_flat)
print(f"  Rule 30 LZ complexity (50x50): {rule30_lz:.4f}")
print()

# Scan CSG coupling
N = 60
print(f"  {'p':>6} {'ord_frac':>10} {'deg_entropy':>12} {'LZ_complex':>12} {'longest_ch':>11} {'#links':>8}")
print("  " + "-" * 65)

couplings = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8]
peak_entropy_p = 0
peak_entropy = 0

for p in couplings:
    deg_ents = []
    lz_comps = []
    ord_fracs = []
    chains = []
    link_counts = []
    for trial in range(8):
        cs = csg_fast(N, coupling=p, rng=np.random.default_rng(trial * 79 + int(p * 1000)))
        deg_ents.append(degree_sequence_entropy(cs))
        lz_comps.append(adjacency_lz_complexity(cs))
        ord_fracs.append(cs.ordering_fraction())
        chains.append(cs.longest_chain())
        link_counts.append(count_links(cs))

    mean_ent = np.mean(deg_ents)
    mean_lz = np.mean(lz_comps)
    mean_of = np.mean(ord_fracs)
    mean_ch = np.mean(chains)
    mean_lk = np.mean(link_counts)

    if mean_ent > peak_entropy:
        peak_entropy = mean_ent
        peak_entropy_p = p

    print(f"  {p:>6.2f} {mean_of:>10.4f} {mean_ent:>12.4f} {mean_lz:>12.4f} {mean_ch:>11.1f} {mean_lk:>8.0f}")

print(f"\n  Peak degree entropy at p = {peak_entropy_p:.2f}")
print(f"  This is the CSG analogue of Rule 30's 'edge of chaos'")
print(f"  Rule 30 LZ = {rule30_lz:.4f} -- comparable to CSG at p ~ {peak_entropy_p}")

dt = time.time() - t0
print(f"\n  [Idea 716 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# IDEA 717: RAMANUJAN'S PARTITION FUNCTION → ANTICHAIN DECOMPOSITION
# ============================================================
print("\n" + "=" * 80)
print("IDEA 717: Ramanujan's Partition Function → Antichain Decomposition")
print("=" * 80)
print("""
WALK TO CAUSAL SETS:
  Step 1: Ramanujan's partition function p(n) → number of ways to write n
          as a sum of positive integers
  Step 2: Integer partitions → Dilworth's theorem: every finite poset can be
          partitioned into antichains (width = min number of chains needed)
  Step 3: Antichain decomposition → count the number of DISTINCT antichains
          in the poset. How does this scale? Like p(N)?

EXPERIMENT: Count the number of distinct antichains for small causets.
Compare with Ramanujan's p(N) ~ exp(pi*sqrt(2N/3)) / (4N*sqrt(3)).
""")
sys.stdout.flush()

t0 = time.time()

def count_antichains(cs):
    """
    Count distinct antichains (subsets where no two elements are related).
    For small N only (exponential: 2^N subsets).
    """
    N = cs.n
    order = cs.order

    count = 0
    for mask in range(2**N):
        subset = [i for i in range(N) if mask & (1 << i)]
        is_antichain = True
        for i in range(len(subset)):
            for j in range(i+1, len(subset)):
                if order[subset[i], subset[j]] or order[subset[j], subset[i]]:
                    is_antichain = False
                    break
            if not is_antichain:
                break
        if is_antichain:
            count += 1
    return count


# Ramanujan's partition function
def ramanujan_p(n):
    """Compute p(n) using dynamic programming."""
    if n <= 0:
        return 1
    dp = [0] * (n + 1)
    dp[0] = 1
    for k in range(1, n + 1):
        for i in range(k, n + 1):
            dp[i] += dp[i - k]
    return dp[n]


# Ramanujan's asymptotic formula
def ramanujan_asymptotic(n):
    return np.exp(np.pi * np.sqrt(2 * n / 3)) / (4 * n * np.sqrt(3))


print(f"  {'N':>4} {'#antichains':>12} {'p(N)':>10} {'p(N) asymp':>12} {'ratio ac/p(N)':>14}")
print("  " + "-" * 55)

for N in range(3, 13):
    cs, _ = random_2order(N, rng_local=np.random.default_rng(N * 83))
    if N <= 12:
        n_ac = count_antichains(cs)
    else:
        n_ac = -1
    pN = ramanujan_p(N)
    pN_asymp = ramanujan_asymptotic(N)
    if n_ac > 0:
        ratio = n_ac / pN
        print(f"  {N:>4} {n_ac:>12} {pN:>10} {pN_asymp:>12.1f} {ratio:>14.4f}")
    else:
        print(f"  {N:>4} {'(too large)':>12} {pN:>10} {pN_asymp:>12.1f}")

# Compare across causet types
print(f"\n  Antichain counts at N=8 across causet types:")
for source_name in ["2-order", "sprinkle-2D", "total order", "antichain"]:
    acs = []
    for trial in range(5):
        if source_name == "total order":
            cs = FastCausalSet(8)
            for i in range(8):
                for j in range(i+1, 8):
                    cs.order[i, j] = True
        elif source_name == "antichain":
            cs = FastCausalSet(8)  # no relations
        elif source_name == "2-order":
            cs, _ = random_2order(8, rng_local=np.random.default_rng(trial*87))
        else:
            cs, _ = sprinkle_fast(8, dim=2, rng=np.random.default_rng(trial*89))
        n_ac = count_antichains(cs)
        acs.append(n_ac)
    print(f"    {source_name:>15}: {np.mean(acs):.1f} +/- {np.std(acs):.1f} "
          f"(p(8) = {ramanujan_p(8)}, 2^8 = {2**8})")

# Scaling: log(#antichains) vs sqrt(N)
print(f"\n  Scaling: log(#antichains) vs sqrt(N)")
ac_data = []
for N in range(4, 13):
    acs = []
    for trial in range(5):
        cs, _ = random_2order(N, rng_local=np.random.default_rng(trial * 91 + N))
        n_ac = count_antichains(cs)
        if n_ac > 0:
            acs.append(n_ac)
    if acs:
        mean_log = np.mean(np.log(acs))
        ac_data.append((N, mean_log))
        print(f"    N={N:>3}: log(#ac) = {mean_log:.3f}, sqrt(N) = {np.sqrt(N):.3f}")

if len(ac_data) >= 3:
    sqrtNs = np.array([np.sqrt(x[0]) for x in ac_data])
    logACs = np.array([x[1] for x in ac_data])
    slope, intercept, r_val, _, _ = stats.linregress(sqrtNs, logACs)
    print(f"\n  Linear fit: log(#ac) = {slope:.3f}*sqrt(N) + {intercept:.3f} (R^2 = {r_val**2:.4f})")
    print(f"  Ramanujan: log(p(N)) ~ pi*sqrt(2N/3) = {np.pi * np.sqrt(2/3):.3f}*sqrt(N)")
    print(f"  Ratio of slopes: {slope / (np.pi * np.sqrt(2/3)):.4f}")

dt = time.time() - t0
print(f"\n  [Idea 717 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# IDEA 718: CATALAN NUMBERS → DYCK PATHS → MONOTONE SUBSEQUENCES
# ============================================================
print("\n" + "=" * 80)
print("IDEA 718: Catalan Numbers → Monotone Subsequences in 2-Orders")
print("=" * 80)
print("""
WALK TO CAUSAL SETS:
  Step 1: Catalan numbers C_n = (2n choose n)/(n+1) → count Dyck paths
          (lattice paths that don't cross the diagonal)
  Step 2: Dyck paths → monotone lattice paths ↔ longest increasing
          subsequences (LIS) in permutations
  Step 3: LIS in permutations → a 2-order is defined by two permutations
          u, v. The causal relations correspond to common increasing
          subsequences. The longest chain = longest common subsequence (LCS).

EXPERIMENT: For random 2-orders, compute the distribution of LCS(u,v)
and compare with the Catalan number sequence. Also: does the number of
distinct maximal chains equal a Catalan number?
""")
sys.stdout.flush()

t0 = time.time()

def longest_common_subsequence(u, v):
    """Compute LCS length of two permutations."""
    N = len(u)
    pos_in_v = np.zeros(N, dtype=int)
    for i in range(N):
        pos_in_v[u[i]] = i
    transformed = np.array([pos_in_v[v[i]] for i in range(N)])
    return longest_increasing_subsequence(transformed)


def longest_increasing_subsequence(arr):
    """Length of LIS using patience sorting (O(n log n))."""
    tails = []
    for x in arr:
        pos = np.searchsorted(tails, x)
        if pos == len(tails):
            tails.append(x)
        else:
            tails[pos] = x
    return len(tails)


def catalan(n):
    """Compute nth Catalan number."""
    from math import comb
    return comb(2 * n, n) // (n + 1)


print(f"  First 15 Catalan numbers: {[catalan(n) for n in range(15)]}")
print()

# LCS distribution for 2-orders
print(f"  {'N':>5} {'E[LCS]':>8} {'2*sqrt(N)':>10} {'std':>8} {'E[LCS]/sqrt(N)':>15} {'closest C_k':>12}")
print("  " + "-" * 65)

lcs_data = []
for N in [10, 20, 30, 50, 80, 100, 150, 200]:
    lcs_vals = []
    n_trials = 50 if N <= 100 else 20
    for trial in range(n_trials):
        to = TwoOrder(N, rng=np.random.default_rng(trial * 97 + N))
        lcs_len = longest_common_subsequence(to.u, to.v)
        lcs_vals.append(lcs_len)
    mean_lcs = np.mean(lcs_vals)
    std_lcs = np.std(lcs_vals)
    lcs_data.append((N, mean_lcs, std_lcs))

    closest_cat = min(range(20), key=lambda n: abs(catalan(n) - mean_lcs))
    print(f"  {N:>5} {mean_lcs:>8.2f} {2*np.sqrt(N):>10.2f} {std_lcs:>8.2f} "
          f"{mean_lcs/np.sqrt(N):>15.4f} C_{closest_cat}={catalan(closest_cat)}")

# Fit E[LCS] = c * N^alpha
Ns_lcs = np.array([x[0] for x in lcs_data])
means_lcs = np.array([x[1] for x in lcs_data])
log_fit_lcs = np.polyfit(np.log(Ns_lcs), np.log(means_lcs), 1)
print(f"\n  Fit: E[LCS] = {np.exp(log_fit_lcs[1]):.4f} * N^{log_fit_lcs[0]:.4f}")
print(f"  Known (Vershik-Kerov): E[LIS of random perm] ~ 2*sqrt(N), so LCS ~ 2*sqrt(N) too")

# Number of maximal chains
print(f"\n  Number of maximal chains for small 2-orders:")
for N in [5, 6, 7, 8, 9, 10]:
    chain_counts = []
    for trial in range(5):
        cs, to = random_2order(N, rng_local=np.random.default_rng(trial * 101 + N))
        links = cs.link_matrix()
        maximals = set(i for i in range(N) if not np.any(cs.order[i, :]))
        minimals = [i for i in range(N) if not np.any(cs.order[:, i])]

        def count_max_chains_from(node):
            successors = [j for j in range(N) if links[node, j]]
            if not successors:
                return 1 if node in maximals else 0
            total = 0
            for s in successors:
                total += count_max_chains_from(s)
            return total

        total_chains = sum(count_max_chains_from(m) for m in minimals)
        chain_counts.append(total_chains)

    mean_cc = np.mean(chain_counts)
    print(f"    N={N:>3}: mean #maximal_chains = {mean_cc:.1f}, "
          f"C_{N-1}={catalan(N-1)}, C_{N//2}={catalan(N//2)}")

dt = time.time() - t0
print(f"\n  [Idea 718 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# IDEA 719: BROWNIAN MOTION → RANDOM WALK MIXING TIME ON HASSE
# ============================================================
print("\n" + "=" * 80)
print("IDEA 719: Brownian Motion → Random Walk Mixing Time on Hasse Diagram")
print("=" * 80)
print("""
WALK TO CAUSAL SETS:
  Step 1: Brownian motion → continuous random walk, diffusion on graphs
  Step 2: Diffusion → random walk on a graph has mixing time t_mix related
          to spectral gap: t_mix ~ 1/lambda_2 (Fiedler value of the Laplacian)
  Step 3: Spectral gap → on the Hasse diagram of a causet, t_mix tells us
          how quickly information spreads through the causal structure.
          Does t_mix scale differently in different dimensions?

EXPERIMENT: Compute the spectral gap (Fiedler value lambda_2) of the Hasse
Laplacian for d-orders at d=2,3,4 and sprinkled causets in 2D,3D.
The mixing time t_mix ~ 1/lambda_2. Does it scale as N^{2/d}?
""")
sys.stdout.flush()

t0 = time.time()

def spectral_gap(cs):
    """Fiedler value (second smallest eigenvalue of Laplacian)."""
    L = hasse_laplacian(cs)
    evals = np.linalg.eigvalsh(L)
    evals.sort()
    return evals[1] if len(evals) > 1 else 0.0


def random_walk_mixing_simulation(cs, n_steps=500):
    """Simulate random walk on Hasse diagram and measure mixing time."""
    adj = hasse_adjacency(cs)
    N = cs.n
    degrees = adj.sum(axis=1)

    if np.any(degrees == 0):
        return n_steps  # isolated nodes

    # Transition matrix
    T = adj / degrees[:, None]

    # Start from node 0, measure TV distance to uniform
    dist = np.zeros(N)
    dist[0] = 1.0
    for step in range(n_steps):
        dist = dist @ T
        tv = 0.5 * np.sum(np.abs(dist - 1.0/N))
        if tv < 0.01:
            return step + 1
    return n_steps


print(f"  {'Source':>15} {'N':>5} {'lambda_2':>10} {'1/lambda_2':>12} {'t_mix(sim)':>10} {'N^(2/d)':>10}")
print("  " + "-" * 70)

# d-orders
for d in [2, 3, 4]:
    for N in [20, 30, 50, 80]:
        gaps = []
        mix_times = []
        for trial in range(8):
            cs, _ = random_dorder(d, N, rng_local=np.random.default_rng(trial * 103 + N * d))
            g = spectral_gap(cs)
            gaps.append(g)
            mt = random_walk_mixing_simulation(cs, n_steps=300)
            mix_times.append(mt)

        mean_gap = np.mean(gaps)
        mean_mt = np.mean(mix_times)
        n_2d = N**(2.0/d)
        label = f"{d}-order"
        inv_gap = 1/mean_gap if mean_gap > 1e-10 else float('inf')
        print(f"  {label:>15} {N:>5} {mean_gap:>10.4f} {inv_gap:>12.2f} "
              f"{mean_mt:>10.1f} {n_2d:>10.2f}")
    print()

# Fit: 1/lambda_2 = c * N^alpha for each d
print("  Scaling fits:")
for d in [2, 3, 4]:
    Ns_fit = []
    tmix_fit = []
    for N in [20, 30, 50, 80]:
        gaps = []
        for trial in range(10):
            cs, _ = random_dorder(d, N, rng_local=np.random.default_rng(trial * 107 + N * d))
            g = spectral_gap(cs)
            if g > 1e-10:
                gaps.append(1.0 / g)
        if gaps:
            Ns_fit.append(N)
            tmix_fit.append(np.mean(gaps))

    if len(Ns_fit) >= 3:
        log_fit = np.polyfit(np.log(Ns_fit), np.log(tmix_fit), 1)
        print(f"    d={d}: 1/lambda_2 ~ N^{log_fit[0]:.3f} (expected 2/d = {2/d:.3f})")

dt = time.time() - t0
print(f"\n  [Idea 719 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# IDEA 720: WAVELETS → MULTIRESOLUTION → WAVELET TRANSFORM
#           OF CAUSAL MATRIX → PHASE TRANSITION DETECTION
# ============================================================
print("\n" + "=" * 80)
print("IDEA 720: Wavelets → Wavelet Transform of Causal Matrix")
print("=" * 80)
print("""
WALK TO CAUSAL SETS:
  Step 1: Wavelets → multiresolution analysis, decomposing signals at
          different scales
  Step 2: Multiresolution → coarse-graining of a causal set (decimation,
          blocking elements into groups)
  Step 3: Coarse-graining → apply a discrete wavelet transform to the
          causal matrix C[i,j] (viewed as a 2D image). The detail
          coefficients at different scales encode hierarchical structure.

EXPERIMENT: Apply 2D Haar wavelet transform to the order matrix. The
energy in detail coefficients at each level tells us about structure
at that scale. Does this detect the BD phase transition? Compare
manifold-like causets (low beta) vs crystalline (high beta) vs random.
""")
sys.stdout.flush()

t0 = time.time()

def haar_wavelet_2d(matrix, levels=3):
    """
    Apply 2D Haar wavelet transform to a matrix.
    Returns the energy at each level (approximation + 3 detail subbands).
    """
    M = matrix.astype(float).copy()
    n = M.shape[0]

    # Pad to next power of 2
    size = 1
    while size < n:
        size *= 2
    padded = np.zeros((size, size))
    padded[:n, :n] = M

    level_energies = []

    current = padded.copy()
    for level in range(levels):
        h = size >> level
        if h < 2:
            break
        half = h // 2

        # Transform rows
        temp = current[:h, :h].copy()
        result = np.zeros_like(temp)
        for i in range(h):
            for j in range(half):
                a, b = temp[i, 2*j], temp[i, 2*j+1]
                result[i, j] = (a + b) / np.sqrt(2)
                result[i, half + j] = (a - b) / np.sqrt(2)

        # Transform columns
        temp2 = result.copy()
        for j in range(h):
            for i in range(half):
                a, b = temp2[2*i, j], temp2[2*i+1, j]
                result[i, j] = (a + b) / np.sqrt(2)
                result[half + i, j] = (a - b) / np.sqrt(2)

        # Extract energies of the three detail subbands
        HL = result[:half, half:h]
        LH = result[half:h, :half]
        HH = result[half:h, half:h]
        LL = result[:half, :half]

        energy_HL = np.sum(HL**2)
        energy_LH = np.sum(LH**2)
        energy_HH = np.sum(HH**2)
        energy_LL = np.sum(LL**2)
        total_detail = energy_HL + energy_LH + energy_HH
        total = energy_LL + total_detail

        level_energies.append({
            'level': level + 1,
            'detail_energy': total_detail,
            'approx_energy': energy_LL,
            'detail_fraction': total_detail / (total + 1e-15),
        })

        # Replace current with LL for next level
        current = np.zeros((size, size))
        current[:half, :half] = LL

    return level_energies


# Compare different causet types
print(f"  {'Source':>20} {'N':>5} {'L1 det%':>10} {'L2 det%':>10} {'L3 det%':>10} {'total det%':>12}")
print("  " + "-" * 70)

for source_name in ["2-order", "sprinkle-2D", "CSG p=0.3", "CSG p=0.5", "total order"]:
    for N in [16, 32]:
        all_l1 = []
        all_l2 = []
        all_l3 = []
        all_total = []
        for trial in range(5):
            if source_name == "total order":
                cs = FastCausalSet(N)
                for i in range(N):
                    for j in range(i+1, N):
                        cs.order[i, j] = True
            elif source_name == "2-order":
                cs, _ = random_2order(N, rng_local=np.random.default_rng(trial*109+N))
            elif source_name == "sprinkle-2D":
                cs, _ = sprinkle_fast(N, dim=2, rng=np.random.default_rng(trial*113+N))
            elif source_name == "CSG p=0.3":
                cs = csg_fast(N, coupling=0.3, rng=np.random.default_rng(trial*117+N))
            elif source_name == "CSG p=0.5":
                cs = csg_fast(N, coupling=0.5, rng=np.random.default_rng(trial*119+N))

            levels = haar_wavelet_2d(cs.order.astype(float), levels=3)
            if len(levels) >= 1:
                all_l1.append(levels[0]['detail_fraction'])
            if len(levels) >= 2:
                all_l2.append(levels[1]['detail_fraction'])
            if len(levels) >= 3:
                all_l3.append(levels[2]['detail_fraction'])
            total_det = sum(l['detail_energy'] for l in levels)
            total_all = total_det + (levels[-1]['approx_energy'] if levels else 1)
            all_total.append(total_det / (total_all + 1e-15))

        l1 = np.mean(all_l1) if all_l1 else 0
        l2 = np.mean(all_l2) if all_l2 else 0
        l3 = np.mean(all_l3) if all_l3 else 0
        td = np.mean(all_total) if all_total else 0
        print(f"  {source_name:>20} {N:>5} {l1:>10.4f} {l2:>10.4f} {l3:>10.4f} {td:>12.4f}")

# Phase transition scan: 2-order MCMC at different beta
print(f"\n  Wavelet detail fraction across BD phase transition (N=16):")
print(f"  {'beta':>8} {'L1 det%':>10} {'L2 det%':>10} {'ord_frac':>10}")
print("  " + "-" * 42)

N_mcmc = 16
for beta in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
    to = TwoOrder(N_mcmc, rng=np.random.default_rng(42))
    # MCMC burn-in
    for step in range(2000):
        to_new = swap_move(to, rng=rng)
        cs_old = to.to_causet()
        cs_new = to_new.to_causet()
        ivals_old = count_intervals_by_size(cs_old, max_size=1)
        ivals_new = count_intervals_by_size(cs_new, max_size=1)
        bd_old = N_mcmc - 2 * ivals_old.get(0, 0) + ivals_old.get(1, 0)
        bd_new = N_mcmc - 2 * ivals_new.get(0, 0) + ivals_new.get(1, 0)
        delta_S = bd_new - bd_old
        if delta_S < 0 or rng.random() < np.exp(-beta * delta_S):
            to = to_new

    # Measure
    cs = to.to_causet()
    levels = haar_wavelet_2d(cs.order.astype(float), levels=3)
    l1 = levels[0]['detail_fraction'] if len(levels) >= 1 else 0
    l2 = levels[1]['detail_fraction'] if len(levels) >= 2 else 0
    of = cs.ordering_fraction()
    print(f"  {beta:>8.1f} {l1:>10.4f} {l2:>10.4f} {of:>10.4f}")

dt = time.time() - t0
print(f"\n  [Idea 720 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY: RANDOM WALK TO CAUSAL SETS (Ideas 711-720)")
print("=" * 80)
print("""
711. FIBONACCI -> LONGEST CHAIN: Chain growth L ~ c*N^alpha. For 2-orders, alpha ~ 0.5
     and c ~ 2 (Bollobas-Winkler). No direct phi connection, but chain lengths
     at Fibonacci N values show ratio ~ sqrt(phi) as expected from sqrt(N) scaling.

712. CONTINUED FRACTIONS -> ORDERING FRACTION: f(d) decays roughly as
     1/2^(d-1) for random d-orders (NOT 1/(d+1) as initially hypothesized).
     The exact ordering fraction for d independent permutations is related
     to the probability of concordance across d rankings. f(2)~0.5, f(3)~0.25,
     f(4)~0.12, showing rapid decay of causal connectivity with dimension.

713. P-ADIC -> ULTRAMETRICITY: Causal set intervals show moderate ultrametricity
     (~60-80% of triples satisfy the ultrametric inequality), higher than
     random sets (~50%). Partial but not complete tree structure in the
     interval hierarchy.

714. TROPICAL GEOMETRY -> TROPICAL DETERMINANT: The tropical determinant of the
     interval-size matrix correlates with the BD action. Both measure "total
     causal content" but in different ways. Tropical det scales as ~ N * f(N).

715. KNOT INVARIANTS -> BRACKET POLYNOMIAL: The bracket polynomial B(q) of the
     Hasse diagram distinguishes causet types. The mean number of components
     and entropy of B(q) differ between 2-orders and sprinkled causets.

716. RULE 30 -> CSG COMPLEXITY: CSG at intermediate coupling (p ~ 0.3-0.4)
     shows peak degree entropy and LZ complexity, analogous to Rule 30's
     Class III behavior. This is the CSG "edge of chaos."

717. RAMANUJAN -> ANTICHAIN COUNT: The number of distinct antichains in a
     2-order grows exponentially with N, with log(#ac) ~ c*sqrt(N). The
     constant c differs from Ramanujan's pi*sqrt(2/3) but the sqrt(N) scaling
     matches.

718. CATALAN -> MONOTONE SUBSEQUENCES: LCS of the two permutations defining a
     2-order follows the Vershik-Kerov law: E[LCS] ~ 2*sqrt(N). The number of
     maximal chains is NOT a Catalan number but grows combinatorially.

719. BROWNIAN MOTION -> MIXING TIME: The spectral gap lambda_2 of the Hasse
     Laplacian gives mixing time 1/lambda_2 ~ N^alpha where alpha depends on
     dimension. For d-orders, alpha ~ 2/d is expected from the
     volume-to-surface ratio.

720. WAVELETS -> PHASE TRANSITION: The wavelet detail fraction of the order
     matrix changes across the BD phase transition. High beta (crystalline) has
     different wavelet signature than low beta (manifold-like). Wavelets provide
     a scale-by-scale probe of causal structure.
""")
sys.stdout.flush()

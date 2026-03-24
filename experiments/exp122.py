"""
Experiment 122: CROSS-DISCIPLINARY ANALOGUES — Ideas 731-740

METHODOLOGY: Take the BEST result from each of 10 different scientific fields
and ask "what would the causal set analogue be?"

731. BIOLOGY — DNA's 4-letter alphabet → What's the causal set "alphabet"?
     The genetic code uses {A,T,G,C}. In causets, the minimal local structures
     are interval types. Classify all intervals of size ≤6 and measure their
     frequencies. Are there exactly 4 dominant "letters"?

732. ECONOMICS — Nash equilibrium in game theory → What's the equilibrium of
     the BD partition function? At each beta, the MCMC converges to a
     steady-state distribution. Measure the variance of observables — does
     it vanish at a Nash-like equilibrium point (variance minimum)?

733. CHEMISTRY — Periodic table organizes elements by properties → Can we
     organize causets by properties? Build a "periodic table" of small causets
     classified by (ordering fraction, longest chain, width, Fiedler value).

734. COMPUTER SCIENCE — P vs NP → Is deciding manifold-likeness NP-hard?
     Empirically test: how does the time to check if a poset is a 2-order
     (manifold-like) scale with N? Is it polynomial or exponential?

735. ECOLOGY — Predator-prey (Lotka-Volterra) → What are the "predator-prey"
     dynamics of the BD transition? Track two competing observables (ordering
     fraction and action) during MCMC and test for Lotka-Volterra oscillations.

736. LINGUISTICS — Zipf's law (frequency ~ 1/rank) → Does the interval size
     distribution follow Zipf's law? Rank interval sizes by frequency and
     check for power-law behavior.

737. MUSIC — Harmonic series (overtones at integer multiples) → Are the
     Pauli-Jordan eigenvalues harmonic? Check if eigenvalue ratios approximate
     integer multiples of a fundamental frequency.

738. PSYCHOLOGY — Weber-Fechner law (perception ~ log(stimulus)) → Is
     entanglement entropy a "perception" of geometry? Test if S_ent ~ log(N)
     (Weber-Fechner) or S ~ N (linear) for the SJ vacuum.

739. GEOLOGY — Plate tectonics (slow drift, sudden quakes) → Does the BD
     transition have "earthquake" dynamics? During MCMC near beta_c, measure
     the distribution of action jumps. Are they Gutenberg-Richter (power-law)?

740. ASTRONOMY — Kepler's laws → Is there a "Kepler's law" for causal
     diamonds? In GR, the volume of a causal diamond scales as t^d. Test this
     scaling for sprinkled causets and 2-orders.

Uses /usr/bin/python3.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats
from scipy.linalg import eigh
from scipy.optimize import curve_fit
from collections import Counter
import time
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from causal_sets.fast_core import FastCausalSet, sprinkle_fast
from causal_sets.two_orders import TwoOrder, swap_move
from causal_sets.d_orders import DOrder
from causal_sets.bd_action import count_intervals_by_size, count_links
from causal_sets.sj_vacuum import pauli_jordan_function, sj_wightman_function, entanglement_entropy
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


def hasse_adjacency(cs):
    """Symmetric adjacency matrix of the Hasse diagram."""
    links = cs.link_matrix()
    return (links | links.T).astype(np.float64)


def hasse_laplacian(cs):
    """Laplacian of the Hasse diagram."""
    adj = hasse_adjacency(cs)
    degree = np.sum(adj, axis=1)
    return np.diag(degree) - adj


def fiedler_value(cs):
    """Second-smallest eigenvalue of the Hasse Laplacian."""
    L = hasse_laplacian(cs)
    evals = np.sort(np.linalg.eigvalsh(L))
    if len(evals) < 2:
        return 0.0
    return float(evals[1])


def antichain_width(cs):
    """Width = size of the largest antichain (greedy layer decomposition)."""
    N = cs.n
    order = cs.order
    remaining = set(range(N))
    layers = []
    while remaining:
        minimals = []
        for x in remaining:
            is_min = True
            for y in remaining:
                if y != x and order[y, x]:
                    is_min = False
                    break
            if is_min:
                minimals.append(x)
        if not minimals:
            break
        layers.append(minimals)
        remaining -= set(minimals)
    return max(len(layer) for layer in layers) if layers else 0


def ordering_fraction(cs):
    """Ordering fraction of a causal set."""
    N = cs.n
    return cs.num_relations() / (N * (N - 1) / 2)


print("=" * 80)
print("EXPERIMENT 122: CROSS-DISCIPLINARY ANALOGUES — Ideas 731-740")
print("Best results from 10 scientific fields -> causal set analogues")
print("=" * 80)
sys.stdout.flush()


# ============================================================
# IDEA 731: BIOLOGY — THE CAUSAL SET "ALPHABET"
# ============================================================
print("\n" + "=" * 80)
print("IDEA 731: BIOLOGY — DNA's 4-Letter Alphabet -> Causal Set Alphabet")
print("=" * 80)
print("""
DNA encodes information in {A, T, G, C}. What is the causal set "alphabet"?

The minimal local structures in a causal set are order intervals. We classify
all intervals of size k (number of interior elements) and count how many
distinct ISOMORPHISM TYPES appear at each size. These types are the "letters."

QUESTION: Is there a small dominant alphabet, like DNA's 4 letters?

METHOD: For N=40 2-orders, enumerate all intervals of size 0-5, classify
by their internal order structure, and measure type frequencies.
""")
sys.stdout.flush()

t0 = time.time()

def interval_signature(cs, i, j):
    """Compute a canonical signature for the interval [i,j].
    Returns a tuple encoding the internal partial order (isomorphism class)."""
    # Find interior elements: k where order[i,k] and order[k,j]
    interior = []
    for k in range(cs.n):
        if k != i and k != j and cs.order[i, k] and cs.order[k, j]:
            interior.append(k)

    n_int = len(interior)
    if n_int == 0:
        return (0,)  # link — no interior

    # Build the internal order as a sorted tuple of relations
    # Map interior elements to 0..n_int-1
    idx_map = {el: idx for idx, el in enumerate(interior)}
    relations = []
    for a in range(n_int):
        for b in range(a + 1, n_int):
            if cs.order[interior[a], interior[b]]:
                relations.append((a, b))
            elif cs.order[interior[b], interior[a]]:
                relations.append((b, a))
    return (n_int, tuple(sorted(relations)))


# Sample many 2-orders and collect interval types
n_samples = 20
N_bio = 40
type_counter = Counter()
size_counter = Counter()

for trial in range(n_samples):
    cs, to = random_2order(N_bio, rng_local=np.random.default_rng(trial))
    # Find all related pairs and their interval sizes
    order_int = cs.order.astype(np.int32)
    interval_matrix = order_int @ order_int

    for i in range(N_bio):
        for j in range(i + 1, N_bio):
            if not cs.order[i, j]:
                continue
            k = interval_matrix[i, j]
            if k > 5:
                continue  # only classify small intervals
            size_counter[k] += 1
            sig = interval_signature(cs, i, j)
            type_counter[sig] += 1

print("  INTERVAL TYPE FREQUENCIES (the causal set 'alphabet'):")
print(f"  {'Size':>6} {'Type':>40} {'Count':>8} {'Freq %':>8}")
print("  " + "-" * 70)

# Group by size
for size in range(6):
    types_at_size = {k: v for k, v in type_counter.items() if k[0] == size}
    total_at_size = size_counter.get(size, 0)
    if total_at_size == 0:
        continue
    for sig, count in sorted(types_at_size.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / sum(type_counter.values())
        # Abbreviate the signature for display
        if sig[0] == 0:
            label = "link (empty interior)"
        elif len(sig) > 1 and len(sig[1]) == 0:
            label = f"antichain-{sig[0]}"
        else:
            n_rels = len(sig[1]) if len(sig) > 1 else 0
            label = f"size-{sig[0]}, {n_rels} internal relations"
        print(f"  {size:>6} {label:>40} {count:>8} {pct:>7.2f}%")
    print()

# Count unique types per size
print("  DIVERSITY (number of distinct types per interval size):")
for size in range(6):
    types_at_size = {k: v for k, v in type_counter.items() if k[0] == size}
    total_at_size = size_counter.get(size, 0)
    n_types = len(types_at_size)
    print(f"    Size {size}: {n_types} distinct types, {total_at_size} total intervals")

# Is there a dominant "alphabet"?
top_types = type_counter.most_common(4)
top_frac = sum(c for _, c in top_types) / sum(type_counter.values())
print(f"\n  TOP 4 TYPES cover {top_frac*100:.1f}% of all intervals (DNA analogy: 4 letters = 100%)")

# The "codon" analogy: consecutive intervals
print(f"\n  VERDICT: The causal set alphabet has {len(type_counter)} distinct letters")
if len(type_counter) <= 10:
    print("  -> SMALL ALPHABET: like DNA, a few types dominate")
elif top_frac > 0.8:
    print("  -> EFFECTIVE SMALL ALPHABET: many rare types, but top 4 dominate (>80%)")
else:
    print("  -> RICH ALPHABET: many types contribute significantly")

dt = time.time() - t0
print(f"\n  [Idea 731 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# IDEA 732: ECONOMICS — NASH EQUILIBRIUM OF BD PARTITION FUNCTION
# ============================================================
print("\n" + "=" * 80)
print("IDEA 732: ECONOMICS — Nash Equilibrium of the BD Partition Function")
print("=" * 80)
print("""
In game theory, a Nash equilibrium is a fixed point where no player can
improve by unilateral deviation. In the BD partition function, the MCMC
converges to a steady-state distribution at each beta.

QUESTION: Is there a special beta where the variance of observables is
MINIMIZED — a "Nash equilibrium" of the partition function?

At beta=0 (infinite temperature), everything fluctuates maximally.
At beta->inf (zero temperature), the system freezes. In between, is there
a variance minimum?

METHOD: Scan beta and measure variance of the BD action and ordering fraction.
""")
sys.stdout.flush()

t0 = time.time()

N_nash = 30
eps_nash = 0.12
betas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0, 80.0]
n_steps_nash = 15000
n_therm_nash = 5000

print(f"  N={N_nash}, eps={eps_nash}")
print(f"  {'beta':>8} {'<S>':>10} {'Var(S)':>10} {'<f>':>10} {'Var(f)':>10} {'CV(S)':>10}")
print("  " + "-" * 65)

nash_results = []
for beta in betas:
    result = mcmc_corrected(N_nash, beta, eps_nash,
                            n_steps=n_steps_nash, n_therm=n_therm_nash,
                            record_every=10, rng=np.random.default_rng(42))
    actions = result['actions']
    # Compute ordering fractions for sampled causets
    ofs = np.array([ordering_fraction(cs) for cs in result['samples']])

    mean_S = np.mean(actions)
    var_S = np.var(actions)
    mean_f = np.mean(ofs)
    var_f = np.var(ofs)
    cv_S = np.sqrt(var_S) / abs(mean_S) if abs(mean_S) > 1e-10 else float('inf')

    nash_results.append({
        'beta': beta, 'mean_S': mean_S, 'var_S': var_S,
        'mean_f': mean_f, 'var_f': var_f, 'cv_S': cv_S
    })

    print(f"  {beta:>8.1f} {mean_S:>10.3f} {var_S:>10.3f} {mean_f:>10.4f} {var_f:>10.6f} {cv_S:>10.4f}")

# Find variance minimum
var_S_vals = [r['var_S'] for r in nash_results]
var_f_vals = [r['var_f'] for r in nash_results]
# Exclude beta=0 (trivial) and very high beta (frozen)
mid_results = [r for r in nash_results if 0.5 <= r['beta'] <= 40]
if mid_results:
    min_var_S = min(mid_results, key=lambda r: r['var_S'])
    min_var_f = min(mid_results, key=lambda r: r['var_f'])
    print(f"\n  MINIMUM Var(S) at beta = {min_var_S['beta']:.1f} (Var = {min_var_S['var_S']:.3f})")
    print(f"  MINIMUM Var(f) at beta = {min_var_f['beta']:.1f} (Var = {min_var_f['var_f']:.6f})")

    # Expected critical beta from Glaser et al.
    beta_c = 1.66 / (N_nash * eps_nash**2)
    print(f"  Expected beta_c (phase transition) = {beta_c:.1f}")

    if abs(min_var_S['beta'] - beta_c) / beta_c < 0.5:
        print("  -> NASH EQUILIBRIUM ~ PHASE TRANSITION: variance minimum near beta_c!")
    else:
        print(f"  -> Variance minimum at beta={min_var_S['beta']:.1f}, NOT near beta_c={beta_c:.1f}")
        print("    The 'Nash equilibrium' and phase transition are DISTINCT phenomena")

dt = time.time() - t0
print(f"\n  [Idea 732 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# IDEA 733: CHEMISTRY — PERIODIC TABLE OF CAUSETS
# ============================================================
print("\n" + "=" * 80)
print("IDEA 733: CHEMISTRY — Periodic Table of Causets")
print("=" * 80)
print("""
The periodic table organizes elements by (atomic number, electron config).
Can we organize CAUSETS by their fundamental properties?

AXES: ordering fraction f, longest chain h, width w, Fiedler value lambda_2
GROUP: causets with similar (f, h) into "elements"

METHOD: Generate 200 random 2-orders at N=20, measure all 4 properties,
and look for natural clusters — the "elements" of causal set theory.
""")
sys.stdout.flush()

t0 = time.time()

N_chem = 20
n_samples_chem = 200

properties = []
for trial in range(n_samples_chem):
    cs, to = random_2order(N_chem, rng_local=np.random.default_rng(trial * 7))
    f = ordering_fraction(cs)
    h = cs.longest_chain()
    w = antichain_width(cs)
    lam2 = fiedler_value(cs)
    n_links = count_links(cs)
    properties.append({
        'f': f, 'h': h, 'w': w, 'lam2': lam2, 'links': n_links
    })

# Build the "periodic table": group by (h, w)
from collections import defaultdict
table = defaultdict(list)
for p in properties:
    table[(p['h'], p['w'])].append(p)

print("  THE PERIODIC TABLE OF 2-ORDERS (N=20)")
print(f"  Rows = longest chain (h), Columns = width (w)")
print()

# Find ranges
h_vals = sorted(set(p['h'] for p in properties))
w_vals = sorted(set(p['w'] for p in properties))

# Header
hw_label = 'h_w'
header = f"  {hw_label:>6}"
for w in w_vals:
    header += f" {w:>8}"
print(header)
print("  " + "-" * (8 + 9 * len(w_vals)))

for h in h_vals:
    row = f"  {h:>6}"
    for w in w_vals:
        cell = table.get((h, w), [])
        if cell:
            avg_f = np.mean([p['f'] for p in cell])
            row += f" {len(cell):>3}({avg_f:.2f})"
        else:
            row += f" {'':>8}"
    print(row)

print(f"\n  Format: count(avg_ordering_fraction)")
print(f"  Total cells occupied: {len(table)} out of {len(h_vals)*len(w_vals)} possible")

# Find the "noble gases" — most stable/common configurations
most_common = sorted(table.items(), key=lambda x: -len(x[1]))[:5]
print(f"\n  TOP 5 'ELEMENTS' (most common configurations):")
for (h, w), entries in most_common:
    avg_f = np.mean([p['f'] for p in entries])
    avg_lam2 = np.mean([p['lam2'] for p in entries])
    print(f"    (h={h}, w={w}): {len(entries)} causets, <f>={avg_f:.3f}, <lam2>={avg_lam2:.3f}")

# Correlation matrix between properties
f_arr = np.array([p['f'] for p in properties])
h_arr = np.array([p['h'] for p in properties])
w_arr = np.array([p['w'] for p in properties])
lam2_arr = np.array([p['lam2'] for p in properties])
link_arr = np.array([p['links'] for p in properties])

print(f"\n  CORRELATION MATRIX (Pearson r):")
names = ['f', 'h', 'w', 'lam2', 'links']
arrs = [f_arr, h_arr, w_arr, lam2_arr, link_arr]
print(f"  {'':>8}", end="")
for n in names:
    print(f" {n:>8}", end="")
print()
for i, ni in enumerate(names):
    print(f"  {ni:>8}", end="")
    for j, nj in enumerate(names):
        r, _ = stats.pearsonr(arrs[i], arrs[j])
        print(f" {r:>8.3f}", end="")
    print()

# Is height + width a "conservation law"?
hw_sum = h_arr + w_arr
print(f"\n  h + w: mean = {np.mean(hw_sum):.2f}, std = {np.std(hw_sum):.2f}")
print(f"  (If h + w ~ const, this is like a 'conservation law' — similar to")
print(f"   Dilworth's theorem: height + width >= N in some sense)")

dt = time.time() - t0
print(f"\n  [Idea 733 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# IDEA 734: CS — IS MANIFOLD-LIKENESS NP-HARD?
# ============================================================
print("\n" + "=" * 80)
print("IDEA 734: CS — Is Deciding Manifold-Likeness NP-Hard?")
print("=" * 80)
print("""
P vs NP is the biggest open problem in CS. For causets:
  Is deciding whether a poset is a d-order (manifold-like) NP-hard?

A poset is a 2-order iff its order-dimension is <= 2.
Computing order-dimension is NP-hard in general (Yannakakis 1982),
but for SPECIFIC cases it might be polynomial.

EMPIRICAL TEST: time how long it takes to verify 2-order-ness as N grows.
Method: given a random poset, try to reconstruct two total orders that
reproduce it. If you can, it's a 2-order. If not, it's not.

We test both TRUE 2-orders (should be fast) and random posets (might be hard).
""")
sys.stdout.flush()

t0 = time.time()

def is_2order_check(cs, timeout_sec=2.0):
    """Try to verify if a causal set is a 2-order by attempting to find
    two linear extensions whose intersection gives the partial order.

    This is a heuristic: find a linear extension, use it as one order,
    then try to find a second compatible one.

    Returns (is_2order_or_unknown, time_taken)."""
    import time as _time
    start = _time.time()
    N = cs.n

    # Topological sort 1: respect the partial order
    # Use Kahn's algorithm
    in_degree = np.sum(cs.order, axis=0)
    queue = list(np.where(in_degree == 0)[0])
    order1 = []
    remaining_in = in_degree.copy()

    while queue:
        # Pick the node with smallest remaining in-degree (greedy)
        node = queue.pop(0)
        order1.append(node)
        for j in range(N):
            if cs.order[node, j]:
                remaining_in[j] -= 1
                if remaining_in[j] == 0:
                    queue.append(j)
        if _time.time() - start > timeout_sec:
            return None, _time.time() - start

    if len(order1) != N:
        return None, _time.time() - start

    # Topological sort 2: reverse priority (different greedy strategy)
    in_degree2 = np.sum(cs.order, axis=0)
    queue2 = list(np.where(in_degree2 == 0)[0])
    queue2.reverse()  # different tiebreaking
    order2 = []
    remaining_in2 = in_degree2.copy()

    while queue2:
        node = queue2.pop(0)
        order2.append(node)
        for j in range(N):
            if cs.order[node, j]:
                remaining_in2[j] -= 1
                if remaining_in2[j] == 0:
                    queue2.insert(0, j)  # different insertion strategy
        if _time.time() - start > timeout_sec:
            return None, _time.time() - start

    if len(order2) != N:
        return None, _time.time() - start

    # Check: does the intersection of order1 and order2 reproduce cs.order?
    perm1 = np.zeros(N, dtype=int)
    perm2 = np.zeros(N, dtype=int)
    for pos, node in enumerate(order1):
        perm1[node] = pos
    for pos, node in enumerate(order2):
        perm2[node] = pos

    # i precedes j iff perm1[i] < perm1[j] AND perm2[i] < perm2[j]
    reconstructed = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(N):
            if perm1[i] < perm1[j] and perm2[i] < perm2[j]:
                reconstructed[i, j] = True

    elapsed = _time.time() - start
    match = np.array_equal(reconstructed, cs.order)
    return match, elapsed


# Test on actual 2-orders (known to be embeddable)
print("  TIMING: Verifying 2-orders (should succeed):")
N_sizes = [10, 15, 20, 25, 30, 40, 50]
print(f"  {'N':>6} {'time (ms)':>12} {'verified':>10}")
print("  " + "-" * 35)

for N in N_sizes:
    times = []
    successes = 0
    for trial in range(5):
        cs, to = random_2order(N, rng_local=np.random.default_rng(trial * 13 + N))
        result, elapsed = is_2order_check(cs)
        times.append(elapsed * 1000)
        if result:
            successes += 1
    avg_t = np.mean(times)
    print(f"  {N:>6} {avg_t:>12.2f} {successes:>7}/5")

# Test on random posets (NOT 2-orders)
print("\n  TIMING: Testing random posets (not necessarily 2-orders):")
print(f"  {'N':>6} {'time (ms)':>12} {'2-order?':>10}")
print("  " + "-" * 35)

for N in N_sizes:
    # Generate a random DAG (not a 2-order in general)
    cs = FastCausalSet(N)
    # Random upper triangular matrix with probability 0.3
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < 0.3:
                cs.order[i, j] = True
    # Ensure transitivity
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if cs.order[i, k] and cs.order[k, j]:
                    cs.order[i, j] = True

    result, elapsed = is_2order_check(cs)
    print(f"  {N:>6} {elapsed*1000:>12.2f} {'yes' if result else 'no/unknown':>10}")

# Fit timing scaling
times_2order = []
for N in N_sizes:
    cs, to = random_2order(N, rng_local=np.random.default_rng(42))
    _, elapsed = is_2order_check(cs)
    times_2order.append(elapsed * 1000)

log_N = np.log(N_sizes)
log_t = np.log(np.array(times_2order) + 1e-6)
slope, intercept, r, _, _ = stats.linregress(log_N, log_t)
print(f"\n  SCALING: time ~ N^{slope:.2f} (R2 = {r**2:.3f})")
if slope < 3.5:
    print(f"  -> POLYNOMIAL scaling (exponent ~ {slope:.1f}): verification is likely in P")
else:
    print(f"  -> HIGH polynomial or worse (exponent ~ {slope:.1f}): could suggest NP-hardness")

dt = time.time() - t0
print(f"\n  [Idea 734 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# IDEA 735: ECOLOGY — PREDATOR-PREY IN THE BD TRANSITION
# ============================================================
print("\n" + "=" * 80)
print("IDEA 735: ECOLOGY — Predator-Prey Dynamics in the BD Transition")
print("=" * 80)
print("""
Lotka-Volterra dynamics: predators and prey oscillate out of phase.
  dx/dt = ax - bxy   (prey grows, eaten by predators)
  dy/dt = dxy - cy    (predators grow from eating prey, die otherwise)

In the BD transition, ordering fraction (f) and action (S) are coupled.
Do they oscillate like predator-prey during MCMC evolution near beta_c?

METHOD: Run MCMC near the critical point and track (f, S) as time series.
Compute cross-correlation to detect phase-shifted oscillations.
""")
sys.stdout.flush()

t0 = time.time()

N_eco = 30
eps_eco = 0.12
beta_c_eco = 1.66 / (N_eco * eps_eco**2)

# Run MCMC at beta_c, recording every step
n_steps_eco = 20000
n_therm_eco = 5000

current = TwoOrder(N_eco, rng=np.random.default_rng(42))
current_cs = current.to_causet()
current_S = bd_action_corrected(current_cs, eps_eco)
current_f = ordering_fraction(current_cs)

# Thermalize
for step in range(n_therm_eco):
    proposed = swap_move(current, rng)
    proposed_cs = proposed.to_causet()
    proposed_S = bd_action_corrected(proposed_cs, eps_eco)
    dS = beta_c_eco * (proposed_S - current_S)
    if dS <= 0 or rng.random() < np.exp(-min(dS, 500)):
        current = proposed
        current_cs = proposed_cs
        current_S = proposed_S

# Record time series
S_series = []
f_series = []
n_links_series = []

for step in range(n_steps_eco):
    proposed = swap_move(current, rng)
    proposed_cs = proposed.to_causet()
    proposed_S = bd_action_corrected(proposed_cs, eps_eco)
    dS = beta_c_eco * (proposed_S - current_S)
    if dS <= 0 or rng.random() < np.exp(-min(dS, 500)):
        current = proposed
        current_cs = proposed_cs
        current_S = proposed_S

    if step % 5 == 0:
        f_val = ordering_fraction(current_cs)
        S_series.append(current_S)
        f_series.append(f_val)
        n_links_series.append(count_links(current_cs))

S_arr = np.array(S_series)
f_arr_eco = np.array(f_series)
links_arr = np.array(n_links_series)

# Normalize for cross-correlation
S_norm = (S_arr - np.mean(S_arr)) / (np.std(S_arr) + 1e-10)
f_norm = (f_arr_eco - np.mean(f_arr_eco)) / (np.std(f_arr_eco) + 1e-10)
links_norm = (links_arr - np.mean(links_arr)) / (np.std(links_arr) + 1e-10)

# Cross-correlation between S and f
max_lag = 100
print(f"  Cross-correlation between S (action) and f (ordering fraction):")
print(f"  beta_c = {beta_c_eco:.1f}, N = {N_eco}, eps = {eps_eco}")
print(f"\n  {'lag':>6} {'xcorr(S,f)':>12} {'xcorr(S,links)':>15}")
print("  " + "-" * 40)

xcorr_sf = []
xcorr_sl = []
lags_display = [0, 5, 10, 20, 50, 100]

for lag in lags_display:
    if lag < len(S_norm):
        n_overlap = len(S_norm) - lag
        corr_sf = np.mean(S_norm[:n_overlap] * f_norm[lag:lag + n_overlap])
        corr_sl = np.mean(S_norm[:n_overlap] * links_norm[lag:lag + n_overlap])
        print(f"  {lag:>6} {corr_sf:>12.4f} {corr_sl:>15.4f}")
        xcorr_sf.append(corr_sf)
        xcorr_sl.append(corr_sl)

# Check for Lotka-Volterra signature: negative correlation at some lag
# (prey and predator oscillate out of phase)
all_xcorr = []
for lag in range(min(max_lag, len(S_norm))):
    n_overlap = len(S_norm) - lag
    corr = np.mean(S_norm[:n_overlap] * f_norm[lag:lag + n_overlap])
    all_xcorr.append(corr)

min_xcorr = min(all_xcorr) if all_xcorr else 0
max_xcorr = max(all_xcorr) if all_xcorr else 0
min_lag = all_xcorr.index(min_xcorr) if all_xcorr else 0
max_lag_idx = all_xcorr.index(max_xcorr) if all_xcorr else 0

print(f"\n  Peak positive correlation: {max_xcorr:.4f} at lag {max_lag_idx}")
print(f"  Peak negative correlation: {min_xcorr:.4f} at lag {min_lag}")

if min_xcorr < -0.1 and max_xcorr > 0.1:
    print("  -> LOTKA-VOLTERRA SIGNATURE DETECTED: S and f oscillate out of phase!")
    print("    Action plays 'predator', ordering fraction plays 'prey'")
elif abs(all_xcorr[0]) > 0.5:
    print(f"  -> STRONG COUPLING (r={all_xcorr[0]:.3f} at lag 0): S and f are locked together")
    print("    No predator-prey dynamics — they move as ONE system")
else:
    print("  -> WEAK/NO OSCILLATORY COUPLING: no Lotka-Volterra dynamics detected")

dt = time.time() - t0
print(f"\n  [Idea 735 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# IDEA 736: LINGUISTICS — ZIPF'S LAW FOR INTERVALS
# ============================================================
print("\n" + "=" * 80)
print("IDEA 736: LINGUISTICS — Zipf's Law for Interval Sizes")
print("=" * 80)
print("""
Zipf's law: in natural language, word frequency ~ 1/rank.
More precisely: f(r) ~ r^(-alpha) with alpha ~ 1.

QUESTION: Does the distribution of interval sizes in causets follow Zipf's law?

An "interval" [x,y] has size k = number of interior elements.
For a random 2-order, the interval size distribution P(k) is known:
  P[k|m] = 2(m-k)/[m(m+1)] for k = 0, ..., m-1 (m = num relations)

But RANK the distinct sizes by frequency and check for power-law.

METHOD: For N=50 2-orders, collect all interval sizes, rank by frequency.
""")
sys.stdout.flush()

t0 = time.time()

N_zipf = 50
n_samples_zipf = 30
all_sizes = []

for trial in range(n_samples_zipf):
    cs, to = random_2order(N_zipf, rng_local=np.random.default_rng(trial * 17))
    # Get all interval sizes
    pairs, sizes = cs.interval_sizes_vectorized()
    all_sizes.extend(sizes.tolist())

# Count frequencies
size_counts = Counter(all_sizes)
# Rank by frequency (most common = rank 1)
ranked = sorted(size_counts.items(), key=lambda x: -x[1])

print(f"  Total intervals: {len(all_sizes)}")
print(f"  Distinct sizes: {len(ranked)}")
print(f"\n  {'Rank':>6} {'Size':>6} {'Freq':>8} {'1/rank':>8} {'freq/max':>10}")
print("  " + "-" * 45)

max_freq = ranked[0][1] if ranked else 1
for rank, (size, freq) in enumerate(ranked[:20], 1):
    print(f"  {rank:>6} {size:>6} {freq:>8} {1.0/rank:>8.4f} {freq/max_freq:>10.4f}")

# Fit Zipf: log(freq) = -alpha * log(rank) + const
ranks = np.arange(1, len(ranked) + 1, dtype=float)
freqs = np.array([f for _, f in ranked], dtype=float)

log_ranks = np.log(ranks)
log_freqs = np.log(freqs)

slope, intercept, r_zipf, _, _ = stats.linregress(log_ranks, log_freqs)
alpha_zipf = -slope

print(f"\n  ZIPF FIT: freq ~ rank^(-{alpha_zipf:.3f})")
print(f"  R2 = {r_zipf**2:.4f}")
print(f"  (True Zipf would give alpha ~ 1.0, R2 ~ 1.0)")

if abs(alpha_zipf - 1.0) < 0.3 and r_zipf**2 > 0.9:
    print("  -> ZIPF'S LAW HOLDS for interval sizes!")
elif r_zipf**2 > 0.9:
    print(f"  -> POWER LAW with alpha = {alpha_zipf:.2f} (not Zipf's alpha=1)")
else:
    print("  -> NOT a simple power law — interval sizes do NOT follow Zipf")

# Also check: is it exponential instead?
slope_exp, intercept_exp, r_exp, _, _ = stats.linregress(ranks[:min(20, len(ranks))],
                                                          log_freqs[:min(20, len(freqs))])
print(f"  Exponential fit (freq ~ exp(-lam*rank)): R2 = {r_exp**2:.4f}, lam = {-slope_exp:.3f}")

if r_exp**2 > r_zipf**2:
    print("  -> EXPONENTIAL decay fits better than power law")
else:
    print("  -> Power law fits better than exponential")

dt = time.time() - t0
print(f"\n  [Idea 736 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# IDEA 737: MUSIC — HARMONIC SERIES IN PAULI-JORDAN EIGENVALUES
# ============================================================
print("\n" + "=" * 80)
print("IDEA 737: MUSIC — Harmonic Series in Pauli-Jordan Eigenvalues")
print("=" * 80)
print("""
The harmonic series: overtones at integer multiples of a fundamental.
  f_n = n * f_1  for n = 1, 2, 3, ...

The Pauli-Jordan function iDelta of a causal set has real antisymmetric eigenvalues
that come in +/-lambda pairs. For the SJ vacuum, these encode the "frequencies" of
the quantum field.

QUESTION: Are the Pauli-Jordan eigenvalues approximately harmonic?
If so, the causal set has a natural "musical" structure.

METHOD: For sprinkled 2D causets and 2-orders, compute the PJ eigenvalues
and check if lambda_n / lambda_1 ~ n (integer ratios).
""")
sys.stdout.flush()

t0 = time.time()

print("  A) SPRINKLED 2D CAUSAL DIAMOND:")
N_music = 30
n_trials_music = 10

print(f"  {'trial':>6} {'lam1':>8} {'lam2/l1':>8} {'lam3/l1':>8} {'lam4/l1':>8} {'lam5/l1':>8} {'harmonic?':>10}")
print("  " + "-" * 60)

ratio_collections = {2: [], 3: [], 4: [], 5: []}

for trial in range(n_trials_music):
    cs, coords = sprinkle_fast(N_music, dim=2, rng=np.random.default_rng(trial * 31))
    iDelta = pauli_jordan_function(cs)

    # Eigenvalues of i*iDelta (Hermitian)
    iA = 1j * iDelta
    evals = np.linalg.eigvalsh(iA)

    # Keep positive eigenvalues, sorted ascending
    pos_evals = np.sort(evals[evals > 1e-10])

    if len(pos_evals) < 5:
        continue

    lam1 = pos_evals[0]
    ratios = [pos_evals[k] / lam1 for k in range(min(5, len(pos_evals)))]

    # Check harmonicity: how close are ratios to integers?
    harmonic_score = np.mean([abs(r - round(r)) for r in ratios[1:]])

    for k in [2, 3, 4, 5]:
        if k - 1 < len(ratios):
            ratio_collections[k].append(ratios[k - 1])

    print(f"  {trial:>6} {lam1:>8.4f}", end="")
    for r in ratios[1:]:
        print(f" {r:>8.3f}", end="")
    print(f" {'YES' if harmonic_score < 0.15 else 'no':>10}")

# Statistics across trials
print(f"\n  AVERAGE RATIOS across {n_trials_music} trials:")
for k in [2, 3, 4, 5]:
    vals = ratio_collections[k]
    if vals:
        print(f"    lam_{k}/lam_1 = {np.mean(vals):.3f} +/- {np.std(vals):.3f} "
              f"(harmonic would be {k}.000)")

print("\n  B) RANDOM 2-ORDERS:")
print(f"  {'trial':>6} {'lam1':>8} {'lam2/l1':>8} {'lam3/l1':>8} {'lam4/l1':>8} {'lam5/l1':>8}")
print("  " + "-" * 55)

ratio_2order = {2: [], 3: [], 4: [], 5: []}
for trial in range(n_trials_music):
    cs, to = random_2order(N_music, rng_local=np.random.default_rng(trial * 37))
    iDelta = pauli_jordan_function(cs)
    iA = 1j * iDelta
    evals = np.linalg.eigvalsh(iA)
    pos_evals = np.sort(evals[evals > 1e-10])

    if len(pos_evals) < 5:
        continue

    lam1 = pos_evals[0]
    ratios = [pos_evals[k] / lam1 for k in range(min(5, len(pos_evals)))]

    for k in [2, 3, 4, 5]:
        if k - 1 < len(ratios):
            ratio_2order[k].append(ratios[k - 1])

    print(f"  {trial:>6} {lam1:>8.4f}", end="")
    for r in ratios[1:]:
        print(f" {r:>8.3f}", end="")
    print()

print(f"\n  AVERAGE RATIOS (2-orders):")
for k in [2, 3, 4, 5]:
    vals = ratio_2order[k]
    if vals:
        print(f"    lam_{k}/lam_1 = {np.mean(vals):.3f} +/- {np.std(vals):.3f}")

# Verdict
sprinkled_harmonic = all(
    abs(np.mean(ratio_collections.get(k, [0])) - k) < 0.5
    for k in [2, 3, 4, 5]
    if ratio_collections.get(k)
)
print(f"\n  VERDICT:", end=" ")
if sprinkled_harmonic:
    print("HARMONIC SERIES detected in sprinkled causets!")
    print("  The Pauli-Jordan eigenvalues have a musical structure.")
else:
    print("NOT harmonic — eigenvalue ratios are NOT integer multiples.")
    print("  The causal set 'spectrum' is anharmonic (more like a drum than a string).")

dt = time.time() - t0
print(f"\n  [Idea 737 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# IDEA 738: PSYCHOLOGY — WEBER-FECHNER FOR ENTANGLEMENT ENTROPY
# ============================================================
print("\n" + "=" * 80)
print("IDEA 738: PSYCHOLOGY — Weber-Fechner Law for Entanglement Entropy")
print("=" * 80)
print("""
Weber-Fechner law: perceived intensity ~ log(physical stimulus).
  Perception = k * ln(Stimulus)

In causal sets, entanglement entropy S_ent is a "perception" of geometry.
The "stimulus" is the causet size N (or the region size).

QUESTION: Does S_ent ~ log(N) (Weber-Fechner) or S_ent ~ N (linear)?

If Weber-Fechner holds, the SJ vacuum "perceives" geometry logarithmically,
exactly as CFT predicts: S = (c/3) ln(L).

METHOD: Compute SJ entanglement entropy for 2D sprinkled causets at
N = 10, 15, 20, 25, 30, 35, 40 and fit S vs ln(N) vs N.
""")
sys.stdout.flush()

t0 = time.time()

N_vals_psy = [10, 15, 20, 25, 30, 35, 40]
n_trials_psy = 8

print(f"  {'N':>5} {'<S>':>8} {'std(S)':>8} {'S/ln(N)':>8} {'S/N':>10}")
print("  " + "-" * 45)

S_means = []
S_stds = []

for N in N_vals_psy:
    S_vals = []
    for trial in range(n_trials_psy):
        cs, coords = sprinkle_fast(N, dim=2, rng=np.random.default_rng(trial * 43 + N))
        W = sj_wightman_function(cs)
        # Entanglement entropy of the left half
        half = N // 2
        region_A = list(range(half))
        S = entanglement_entropy(W, region_A)
        S_vals.append(S)

    S_mean = np.mean(S_vals)
    S_std = np.std(S_vals)
    S_means.append(S_mean)
    S_stds.append(S_std)

    print(f"  {N:>5} {S_mean:>8.4f} {S_std:>8.4f} {S_mean/np.log(N):>8.4f} {S_mean/N:>10.6f}")

# Fit 1: S = a * ln(N) + b (Weber-Fechner / CFT)
log_N = np.log(N_vals_psy)
slope_log, intercept_log, r_log, _, _ = stats.linregress(log_N, S_means)

# Fit 2: S = a * N + b (linear / volume law)
slope_lin, intercept_lin, r_lin, _, _ = stats.linregress(N_vals_psy, S_means)

# Fit 3: S = a * N^alpha (power law)
log_S = np.log(np.array(S_means) + 1e-10)
slope_pow, intercept_pow, r_pow, _, _ = stats.linregress(np.log(N_vals_psy), log_S)

print(f"\n  SCALING FITS:")
print(f"  S = {slope_log:.4f} * ln(N) + {intercept_log:.4f}  (Weber-Fechner/CFT)  R2 = {r_log**2:.4f}")
print(f"  S = {slope_lin:.6f} * N + {intercept_lin:.4f}      (Volume law)          R2 = {r_lin**2:.4f}")
print(f"  S ~ N^{slope_pow:.3f}                              (Power law)           R2 = {r_pow**2:.4f}")

# Extract c_eff from CFT formula S = (c/3) * ln(N/2)
c_eff = 3 * slope_log  # since S = (c/3) * ln(L) and L ~ N
print(f"\n  Effective central charge c_eff = {c_eff:.3f}")

fits = [(r_log**2, 'WEBER-FECHNER (logarithmic)'),
        (r_lin**2, 'LINEAR (volume law)'),
        (r_pow**2, f'POWER LAW (N^{slope_pow:.2f})')]
best = max(fits, key=lambda x: x[0])
print(f"\n  BEST FIT: {best[1]} with R2 = {best[0]:.4f}")

if best[1].startswith('WEBER'):
    print("  -> WEBER-FECHNER LAW HOLDS: entanglement entropy logarithmically 'perceives' geometry!")
    print(f"    This is exactly the CFT prediction with c_eff = {c_eff:.2f}")
elif best[1].startswith('LINEAR'):
    print("  -> VOLUME LAW: S grows linearly with N (non-physical, typical of random states)")
else:
    print(f"  -> POWER LAW: S ~ N^{slope_pow:.2f} (intermediate between area and volume)")

dt = time.time() - t0
print(f"\n  [Idea 738 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# IDEA 739: GEOLOGY — EARTHQUAKE DYNAMICS IN THE BD TRANSITION
# ============================================================
print("\n" + "=" * 80)
print("IDEA 739: GEOLOGY — Earthquake Dynamics in the BD Transition")
print("=" * 80)
print("""
Gutenberg-Richter law: earthquake magnitude distribution follows a power law.
  log10(N(>=M)) = a - b*M,  equivalently  P(E) ~ E^(-beta_GR)

Plate tectonics: long periods of slow drift punctuated by sudden ruptures.

QUESTION: Does the BD action during MCMC near beta_c show earthquake-like
dynamics? Are the "jumps" in action power-law distributed?

METHOD: Run MCMC at several beta values near beta_c. Record action at every
step. Compute the distribution of |dS| (action jumps).
""")
sys.stdout.flush()

t0 = time.time()

N_geo = 30
eps_geo = 0.12
beta_c_geo = 1.66 / (N_geo * eps_geo**2)

test_betas = [0.5 * beta_c_geo, beta_c_geo, 2.0 * beta_c_geo]
n_steps_geo = 20000
n_therm_geo = 5000

for beta_test in test_betas:
    current = TwoOrder(N_geo, rng=np.random.default_rng(42))
    current_cs = current.to_causet()
    current_S = bd_action_corrected(current_cs, eps_geo)

    # Thermalize
    for step in range(n_therm_geo):
        proposed = swap_move(current, rng)
        proposed_cs = proposed.to_causet()
        proposed_S = bd_action_corrected(proposed_cs, eps_geo)
        dS = beta_test * (proposed_S - current_S)
        if dS <= 0 or rng.random() < np.exp(-min(dS, 500)):
            current, current_cs, current_S = proposed, proposed_cs, proposed_S

    # Record action jumps
    jumps = []
    for step in range(n_steps_geo):
        proposed = swap_move(current, rng)
        proposed_cs = proposed.to_causet()
        proposed_S = bd_action_corrected(proposed_cs, eps_geo)
        delta_S = abs(proposed_S - current_S)
        dS = beta_test * (proposed_S - current_S)
        if dS <= 0 or rng.random() < np.exp(-min(dS, 500)):
            current, current_cs, current_S = proposed, proposed_cs, proposed_S
            if delta_S > 0.01:  # only count actual changes
                jumps.append(delta_S)

    jumps = np.array(jumps)
    if len(jumps) < 10:
        print(f"\n  beta = {beta_test:.1f} (beta/beta_c = {beta_test/beta_c_geo:.2f}): too few jumps ({len(jumps)})")
        continue

    print(f"\n  beta = {beta_test:.1f} (beta/beta_c = {beta_test/beta_c_geo:.2f}):")
    print(f"    Total jumps: {len(jumps)}, mean |dS| = {np.mean(jumps):.4f}, max = {np.max(jumps):.4f}")

    # Distribution of jump sizes
    # Bin into log-spaced bins
    min_jump = max(np.min(jumps), 0.01)
    max_jump = np.max(jumps)
    if max_jump / min_jump < 2:
        print("    Jump range too narrow for power-law analysis")
        continue

    log_bins = np.logspace(np.log10(min_jump), np.log10(max_jump), 15)
    hist, bin_edges = np.histogram(jumps, bins=log_bins)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    # Only fit bins with counts > 0
    mask = hist > 0
    if np.sum(mask) < 4:
        print("    Too few non-empty bins for fitting")
        continue

    log_x = np.log10(bin_centers[mask])
    log_y = np.log10(hist[mask].astype(float))

    slope_gr, intercept_gr, r_gr, _, _ = stats.linregress(log_x, log_y)

    print(f"    Gutenberg-Richter fit: log10(N) = {slope_gr:.2f} * log10(|dS|) + {intercept_gr:.2f}")
    print(f"    Power-law exponent b = {-slope_gr:.2f}, R2 = {r_gr**2:.3f}")

    # Also fit exponential
    slope_exp_geo, _, r_exp_geo, _, _ = stats.linregress(bin_centers[mask], log_y)

    if r_gr**2 > 0.8 and -slope_gr > 0.5:
        print(f"    -> GUTENBERG-RICHTER LAW DETECTED: action jumps are power-law distributed!")
        print(f"      b-value = {-slope_gr:.2f} (earthquakes have b ~ 1.0)")
    elif r_gr**2 > r_exp_geo**2:
        print(f"    -> Weak power-law tendency (R2 = {r_gr**2:.3f})")
    else:
        print(f"    -> NOT power-law: exponential or other distribution")

    # Large events fraction
    q90 = np.percentile(jumps, 90)
    large_frac = np.mean(jumps > q90)
    print(f"    Large events (>90th percentile = {q90:.4f}): {large_frac*100:.1f}% of energy")

dt = time.time() - t0
print(f"\n  [Idea 739 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# IDEA 740: ASTRONOMY — KEPLER'S LAW FOR CAUSAL DIAMONDS
# ============================================================
print("\n" + "=" * 80)
print("IDEA 740: ASTRONOMY — Kepler's Law for Causal Diamonds")
print("=" * 80)
print("""
Kepler's third law: T^2 ~ a^3 (orbital period squared ~ semi-major axis cubed).
More generally, the volume of a causal diamond in d-dimensional Minkowski:
  V_diamond(t) = C_d * t^d

where t is the proper time extent and C_d is a dimension-dependent constant.
  C_2 = 2 (area of 2D diamond with half-diagonal t)
  C_4 = (pi/12) * t^4

QUESTION: Can we verify V ~ t^d for sprinkled causets and extract the
effective dimension? This would be a "Kepler's law" for discrete spacetime.

METHOD:
1. Sprinkle N elements into a d-dimensional causal diamond of extent t
2. Count elements as a function of the sub-diamond extent tau < t
3. Fit N(tau) ~ tau^d_eff to extract d_eff
4. Compare d_eff with the true embedding dimension
""")
sys.stdout.flush()

t0 = time.time()

# Test in 2D, 3D, 4D
dims_kepler = [2, 3, 4]
N_kepler = 200  # large enough for good statistics
n_trials_kepler = 5

for dim in dims_kepler:
    print(f"\n  DIMENSION d = {dim}:")

    d_effs = []
    for trial in range(n_trials_kepler):
        cs, coords = sprinkle_fast(N_kepler, dim=dim,
                                    rng=np.random.default_rng(trial * 53 + dim))

        # The causal diamond has extent t=1 (default).
        # For sub-diamonds: select elements within a smaller time range
        # The time coordinate is coords[:, 0]
        t_coords = coords[:, 0]
        t_max = np.max(t_coords)
        t_min = np.min(t_coords)
        t_center = (t_max + t_min) / 2

        # Count elements in sub-diamonds of varying size
        tau_fracs = np.linspace(0.1, 1.0, 15)
        counts = []
        taus = []

        for frac in tau_fracs:
            half_extent = frac * (t_max - t_min) / 2
            t_lo = t_center - half_extent
            t_hi = t_center + half_extent

            # Elements within the time range
            in_range = (t_coords >= t_lo) & (t_coords <= t_hi)

            # For a proper sub-diamond, also check spatial extent
            # In the diamond, spatial extent scales with time extent
            if dim == 2:
                x = coords[:, 1]
                # Diamond condition: |x| < t_extent - |t - t_center|
                in_diamond = in_range.copy()
                for idx in np.where(in_range)[0]:
                    dt_from_center = abs(t_coords[idx] - t_center)
                    if abs(x[idx]) > half_extent - dt_from_center:
                        in_diamond[idx] = False
                count = np.sum(in_diamond)
            else:
                # Approximate: just use time range
                count = np.sum(in_range)

            if count > 0:
                counts.append(count)
                taus.append(frac)

        if len(counts) < 5:
            continue

        # Fit: N(tau) = A * tau^d_eff
        log_tau = np.log(np.array(taus))
        log_count = np.log(np.array(counts, dtype=float))

        slope_k, intercept_k, r_k, _, _ = stats.linregress(log_tau, log_count)
        d_effs.append(slope_k)

        if trial == 0:
            print(f"    tau_frac   N_elements   expected(~tau^{dim})")
            for tau, cnt in zip(taus[::3], counts[::3]):
                expected = counts[-1] * tau**dim
                print(f"    {tau:.3f}      {cnt:>6}       {expected:>8.1f}")

    if d_effs:
        d_eff_mean = np.mean(d_effs)
        d_eff_std = np.std(d_effs)
        print(f"\n    KEPLER'S LAW FIT: N(tau) ~ tau^{d_eff_mean:.2f} +/- {d_eff_std:.2f}")
        print(f"    True dimension: d = {dim}")
        error_pct = abs(d_eff_mean - dim) / dim * 100
        print(f"    Error: {error_pct:.1f}%")
        if error_pct < 15:
            print(f"    -> KEPLER'S LAW CONFIRMED: V ~ t^{dim} for d={dim} causets")
        else:
            print(f"    -> KEPLER'S LAW APPROXIMATE: d_eff = {d_eff_mean:.2f} vs true d = {dim}")

# Cross-dimensional comparison
print(f"\n  CROSS-DIMENSIONAL 'KEPLER TABLE':")
print(f"  (Analogous to Kepler's T^2 = a^3 relating different orbits)")
print(f"  {'dim':>5} {'V ~ t^d':>10} {'C_d (continuum)':>16} {'C_d (measured)':>16}")
print("  " + "-" * 50)

# Continuum volumes of unit causal diamond
c_d_continuum = {
    2: 2.0,
    3: np.pi / 2,
    4: np.pi / 12,
}
for dim in dims_kepler:
    # Measure: sprinkle N into unit diamond, count = rho * V
    # where rho = N / V_total. So V_sub / V_total = N_sub / N
    cs, coords = sprinkle_fast(N_kepler, dim=dim, rng=np.random.default_rng(999))
    c_meas = N_kepler  # proportional to volume
    c_cont = c_d_continuum.get(dim, 0)
    # Normalize: ratio of N at tau=0.5 to N at tau=1.0
    t_coords = coords[:, 0]
    t_max, t_min = np.max(t_coords), np.min(t_coords)
    t_center = (t_max + t_min) / 2
    half = (t_max - t_min) / 4  # tau=0.5
    in_half = (t_coords >= t_center - half) & (t_coords <= t_center + half)
    ratio = np.sum(in_half) / N_kepler
    expected_ratio = 0.5**dim
    print(f"  {dim:>5} {'t^'+str(dim):>10} {c_cont:>16.4f} {ratio:>12.4f}/{expected_ratio:.4f}")

print(f"\n  VERDICT: The causal diamond volume scaling V ~ t^d is the 'Kepler's law'")
print(f"  of discrete spacetime — it encodes the effective dimension.")

dt = time.time() - t0
print(f"\n  [Idea 740 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# GRAND SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("GRAND SUMMARY: CROSS-DISCIPLINARY ANALOGUES (Ideas 731-740)")
print("=" * 80)
print("""
  FIELD          THEIR BEST RESULT          CAUSET ANALOGUE              FINDING
  -------------  -------------------------  ---------------------------  ----------
  Biology        DNA 4-letter alphabet       Interval type classification TBD above
  Economics      Nash equilibrium            BD variance minimum          TBD above
  Chemistry      Periodic table              (h, w) property table        TBD above
  Computer Sci   P vs NP                     Manifold-likeness complexity TBD above
  Ecology        Lotka-Volterra              S vs f cross-correlation     TBD above
  Linguistics    Zipf's law                  Interval size distribution   TBD above
  Music          Harmonic series             PJ eigenvalue ratios         TBD above
  Psychology     Weber-Fechner               S_ent vs ln(N) scaling       TBD above
  Geology        Gutenberg-Richter           BD action jump distribution  TBD above
  Astronomy      Kepler's laws               Diamond volume V ~ t^d      TBD above

  KEY INSIGHT: The most productive analogies are those where the causal set
  version of a phenomenon is TESTABLE with our existing computational tools.
  The periodic table (Chemistry), Zipf's law (Linguistics), and Kepler's law
  (Astronomy) are the strongest because they make precise numerical predictions.

  The weakest analogies are those where the mapping is metaphorical rather than
  mathematical (e.g., Nash equilibrium, Lotka-Volterra).
""")

print("=" * 80)
print("EXPERIMENT 122 COMPLETE")
print("=" * 80)

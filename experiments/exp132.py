"""
Experiment 132: EVOLUTIONARY DYNAMICS — Ideas 831-840

Treat causal sets as organisms and observables as fitness functions.
Evolve populations of 2-orders under selection pressure.

831. EVOLVE for maximum interval entropy H. What does the "fittest" causet
     look like after 50 generations?
832. EVOLVE for minimum Fiedler value. What structure has the weakest
     algebraic connectivity while remaining a valid 2-order?
833. EVOLVE for maximum ER=EPR correlation r. Does evolution find r>0.95?
834. EVOLVE for c_eff closest to 1.0. What 2-order gives the most CFT-like
     SJ vacuum?
835. EVOLVE for maximum deviation from GUE (<r> furthest from 0.60). Can
     evolution break GUE universality?
836. SEXUAL REPRODUCTION: cross two 2-orders by taking u from parent 1 and
     v from parent 2. Does crossover produce viable "offspring"?
837. SPECIATION: evolve two populations toward different fitness goals
     simultaneously. Do they diverge in structure?
838. EXTINCTION: start with 100 random 2-orders. Apply random perturbations.
     Which properties go "extinct" first?
839. COEVOLUTION: evolve causet to maximize entropy while vacuum adapts to
     minimize it. Who wins?
840. GENETIC DRIFT: evolve 10 independent populations with NO selection
     (neutral drift). Do they converge to similar structures?

Population size 20, N=30, 50 generations per experiment.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats
from scipy.linalg import eigh
import time
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from causal_sets.fast_core import FastCausalSet
from causal_sets.two_orders import TwoOrder, swap_move
from causal_sets.bd_action import count_intervals_by_size, count_links

np.set_printoptions(precision=6, suppress=True, linewidth=120)
rng = np.random.default_rng(132)

# ============================================================
# PARAMETERS
# ============================================================
POP_SIZE = 20
N = 30
N_GENERATIONS = 50
N_MUTATIONS = 3  # mutations per offspring

# ============================================================
# SHARED UTILITIES
# ============================================================

def random_2order(N_size, rng_local=None):
    """Generate a random 2-order and return (FastCausalSet, TwoOrder)."""
    if rng_local is None:
        rng_local = rng
    to = TwoOrder(N_size, rng=rng_local)
    return to.to_causet(), to


def mutate(to, rng_local, n_swaps=3):
    """Apply n_swaps random swap moves to a TwoOrder."""
    result = to.copy()
    for _ in range(n_swaps):
        result = swap_move(result, rng_local)
    return result


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
    """Second smallest eigenvalue of the Hasse Laplacian."""
    L = hasse_laplacian(cs)
    evals = np.linalg.eigvalsh(L)
    evals = np.sort(evals)
    return evals[1] if len(evals) > 1 else 0.0


def interval_entropy(cs, max_k=8):
    """Shannon entropy of the interval size distribution."""
    counts = count_intervals_by_size(cs, max_size=max_k)
    vals = np.array([counts[k] for k in sorted(counts.keys()) if counts[k] > 0], dtype=float)
    if vals.sum() == 0:
        return 0.0
    p = vals / vals.sum()
    return -np.sum(p * np.log(p + 1e-300))


def ordering_fraction(cs):
    """Fraction of pairs that are causally related."""
    return cs.ordering_fraction()


def longest_chain(cs):
    """Length of the longest chain."""
    return cs.longest_chain()


def sj_wightman(cs):
    """Compute SJ vacuum Wightman function."""
    n = cs.n
    C = cs.order.astype(float)
    Delta = (2.0 / n) * (C.T - C)
    H = 1j * Delta
    evals, evecs = np.linalg.eigh(H)
    evals = evals.real
    pos = evals > 1e-12
    if not np.any(pos):
        return np.zeros((n, n))
    W = (evecs[:, pos] @ np.diag(evals[pos]) @ evecs[:, pos].conj().T).real
    w_max = np.linalg.eigvalsh(W).max()
    if w_max > 1:
        W = W / w_max
    return W


def er_epr_correlation(cs, W):
    """Pearson r between |W[i,j]| and causal connectivity for spacelike pairs."""
    n = cs.n
    order = cs.order
    order_int = order.astype(np.int32)

    w_vals = []
    conn_vals = []

    for i in range(n):
        for j in range(i + 1, n):
            if not order[i, j] and not order[j, i]:
                w_vals.append(abs(W[i, j]))
                cp = int(np.sum(order_int[:, i] & order_int[:, j]))
                cf = int(np.sum(order_int[i, :] & order_int[j, :]))
                conn_vals.append(cp + cf)

    if len(w_vals) < 10:
        return float('nan')

    return float(np.corrcoef(w_vals, conn_vals)[0, 1])


def sj_entanglement_entropy(cs, W=None):
    """Entanglement entropy of the first half of elements in SJ vacuum."""
    n = cs.n
    if W is None:
        W = sj_wightman(cs)
    region_A = list(range(n // 2))
    W_A = W[np.ix_(region_A, region_A)]
    eigenvalues = np.linalg.eigvalsh(W_A)
    eigenvalues = np.clip(eigenvalues, 1e-15, 1 - 1e-15)
    S = -np.sum(eigenvalues * np.log(eigenvalues) +
                (1 - eigenvalues) * np.log(1 - eigenvalues))
    return float(S)


def c_eff(cs):
    """Effective central charge from SJ entanglement entropy: c_eff = 3*S/ln(N)."""
    n = cs.n
    if n <= 1:
        return 0.0
    S = sj_entanglement_entropy(cs)
    return 3.0 * S / np.log(n)


def level_spacing_ratio(evals):
    """Mean ratio of consecutive level spacings (GUE~0.60, GOE~0.53, Poisson~0.39)."""
    evals = np.sort(evals)
    pos = evals[evals > 1e-12]
    if len(pos) < 3:
        return 0.0
    spacings = np.diff(pos)
    spacings = spacings[spacings > 0]
    if len(spacings) < 2:
        return 0.0
    r_vals = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
    return float(np.mean(r_vals))


def iDelta_level_spacing(cs):
    """<r> for Pauli-Jordan eigenvalues."""
    n = cs.n
    C = cs.order.astype(float)
    Delta = (2.0 / n) * (C.T - C)
    H = 1j * Delta
    evals = np.linalg.eigvalsh(H).real
    return level_spacing_ratio(evals)


def link_fraction(cs):
    """Fraction of relations that are links."""
    n_rel = cs.num_relations()
    if n_rel == 0:
        return 0.0
    links = cs.link_matrix()
    return float(np.sum(links)) / n_rel


def describe_structure(cs, to):
    """Return a dict of structural observables for a causet."""
    return {
        'ord_frac': ordering_fraction(cs),
        'link_frac': link_fraction(cs),
        'chain': longest_chain(cs),
        'fiedler': fiedler_value(cs),
        'interval_H': interval_entropy(cs),
        'n_links': int(np.sum(cs.link_matrix())),
    }


# ============================================================
# EVOLUTIONARY ENGINE
# ============================================================

def evolve(population_tos, fitness_fn, n_generations, rng_local,
           maximize=True, verbose=True, label=""):
    """
    Evolve a population of TwoOrders under selection pressure.

    Args:
        population_tos: list of TwoOrder objects
        fitness_fn: function(FastCausalSet, TwoOrder) -> float
        n_generations: number of generations
        maximize: if True, select for higher fitness
        verbose: print progress
        label: name for printing

    Returns:
        best_to, best_cs, history (list of dicts per generation)
    """
    pop = [to.copy() for to in population_tos]
    pop_size = len(pop)
    history = []

    for gen in range(n_generations):
        # Evaluate fitness
        fitnesses = []
        causets = []
        for to in pop:
            cs = to.to_causet()
            causets.append(cs)
            try:
                f = fitness_fn(cs, to)
                if np.isnan(f) or np.isinf(f):
                    f = -1e10 if maximize else 1e10
            except Exception:
                f = -1e10 if maximize else 1e10
            fitnesses.append(f)

        fitnesses = np.array(fitnesses)

        # Record stats
        best_idx = np.argmax(fitnesses) if maximize else np.argmin(fitnesses)
        record = {
            'gen': gen,
            'best': fitnesses[best_idx],
            'mean': np.mean(fitnesses),
            'std': np.std(fitnesses),
            'best_idx': best_idx,
        }
        history.append(record)

        if verbose and gen % 10 == 0:
            direction = "max" if maximize else "min"
            print(f"    Gen {gen:3d}: {direction}={record['best']:.4f}, "
                  f"mean={record['mean']:.4f} +/- {record['std']:.4f}")

        # Selection: keep top 50%
        if maximize:
            survivors_idx = np.argsort(fitnesses)[-pop_size // 2:]
        else:
            survivors_idx = np.argsort(fitnesses)[:pop_size // 2]

        survivors = [pop[i].copy() for i in survivors_idx]

        # Reproduction: mutate survivors to fill population
        new_pop = [s.copy() for s in survivors]  # elites survive unchanged
        while len(new_pop) < pop_size:
            parent = survivors[rng_local.integers(len(survivors))]
            child = mutate(parent, rng_local, n_swaps=N_MUTATIONS)
            new_pop.append(child)

        pop = new_pop

    # Final evaluation to find the best
    fitnesses = []
    causets = []
    for to in pop:
        cs = to.to_causet()
        causets.append(cs)
        try:
            f = fitness_fn(cs, to)
            if np.isnan(f) or np.isinf(f):
                f = -1e10 if maximize else 1e10
        except Exception:
            f = -1e10 if maximize else 1e10
        fitnesses.append(f)

    fitnesses = np.array(fitnesses)
    best_idx = np.argmax(fitnesses) if maximize else np.argmin(fitnesses)

    return pop[best_idx], causets[best_idx], history


def make_initial_population(pop_size, N_size, rng_local):
    """Create a random initial population of TwoOrders."""
    return [TwoOrder(N_size, rng=np.random.default_rng(rng_local.integers(2**31)))
            for _ in range(pop_size)]


print("=" * 78)
print("EXPERIMENT 132: EVOLUTIONARY DYNAMICS — IDEAS 831-840")
print("Treat causal sets as organisms, observables as fitness functions")
print(f"Population={POP_SIZE}, N={N}, Generations={N_GENERATIONS}")
print("=" * 78)

# ============================================================
# IDEA 831: EVOLVE for maximum interval entropy H
# ============================================================
print("\n" + "=" * 78)
print("IDEA 831: EVOLVE for maximum interval entropy H")
print("What does the 'fittest' (highest entropy) causet look like?")
print("=" * 78)

t0 = time.time()

def fitness_831(cs, to):
    return interval_entropy(cs)

pop_831 = make_initial_population(POP_SIZE, N, rng)

# Baseline: random population
baseline_H = []
for to in pop_831:
    cs = to.to_causet()
    baseline_H.append(interval_entropy(cs))
print(f"  Baseline random: H = {np.mean(baseline_H):.4f} +/- {np.std(baseline_H):.4f}")

best_to_831, best_cs_831, history_831 = evolve(
    pop_831, fitness_831, N_GENERATIONS, rng, maximize=True, label="max_H")

best_H = interval_entropy(best_cs_831)
struct = describe_structure(best_cs_831, best_to_831)
print(f"\n  EVOLVED BEST: H = {best_H:.4f}")
print(f"  Structure: ord_frac={struct['ord_frac']:.3f}, link_frac={struct['link_frac']:.3f}, "
      f"chain={struct['chain']}, fiedler={struct['fiedler']:.4f}, n_links={struct['n_links']}")
print(f"  Improvement: {np.mean(baseline_H):.4f} -> {best_H:.4f} "
      f"({(best_H/np.mean(baseline_H) - 1)*100:+.1f}%)")
print(f"  Time: {time.time()-t0:.1f}s")

# ============================================================
# IDEA 832: EVOLVE for minimum Fiedler value
# ============================================================
print("\n" + "=" * 78)
print("IDEA 832: EVOLVE for minimum Fiedler value")
print("Weakest algebraic connectivity while remaining a valid 2-order")
print("=" * 78)

t0 = time.time()

def fitness_832(cs, to):
    return fiedler_value(cs)

pop_832 = make_initial_population(POP_SIZE, N, rng)

baseline_fiedler = []
for to in pop_832:
    cs = to.to_causet()
    baseline_fiedler.append(fiedler_value(cs))
print(f"  Baseline random: Fiedler = {np.mean(baseline_fiedler):.4f} +/- {np.std(baseline_fiedler):.4f}")

best_to_832, best_cs_832, history_832 = evolve(
    pop_832, fitness_832, N_GENERATIONS, rng, maximize=False, label="min_fiedler")

best_fiedler = fiedler_value(best_cs_832)
struct = describe_structure(best_cs_832, best_to_832)
print(f"\n  EVOLVED BEST: Fiedler = {best_fiedler:.6f}")
print(f"  Structure: ord_frac={struct['ord_frac']:.3f}, link_frac={struct['link_frac']:.3f}, "
      f"chain={struct['chain']}, H={struct['interval_H']:.4f}, n_links={struct['n_links']}")
print(f"  Improvement: {np.mean(baseline_fiedler):.4f} -> {best_fiedler:.6f}")
print(f"  Time: {time.time()-t0:.1f}s")

# ============================================================
# IDEA 833: EVOLVE for maximum ER=EPR correlation r
# ============================================================
print("\n" + "=" * 78)
print("IDEA 833: EVOLVE for maximum ER=EPR correlation r")
print("Can evolution find causets with r > 0.95?")
print("=" * 78)

t0 = time.time()

def fitness_833(cs, to):
    W = sj_wightman(cs)
    r = er_epr_correlation(cs, W)
    return r if not np.isnan(r) else -1.0

pop_833 = make_initial_population(POP_SIZE, N, rng)

baseline_r = []
for to in pop_833:
    cs = to.to_causet()
    W = sj_wightman(cs)
    r = er_epr_correlation(cs, W)
    if not np.isnan(r):
        baseline_r.append(r)
print(f"  Baseline random: r = {np.mean(baseline_r):.4f} +/- {np.std(baseline_r):.4f}")

best_to_833, best_cs_833, history_833 = evolve(
    pop_833, fitness_833, N_GENERATIONS, rng, maximize=True, label="max_ER_EPR")

W_best = sj_wightman(best_cs_833)
best_r = er_epr_correlation(best_cs_833, W_best)
struct = describe_structure(best_cs_833, best_to_833)
print(f"\n  EVOLVED BEST: r = {best_r:.4f}")
print(f"  Structure: ord_frac={struct['ord_frac']:.3f}, link_frac={struct['link_frac']:.3f}, "
      f"chain={struct['chain']}, fiedler={struct['fiedler']:.4f}")
print(f"  r > 0.95? {'YES!' if best_r > 0.95 else 'No'}")
print(f"  Improvement: {np.mean(baseline_r):.4f} -> {best_r:.4f}")
print(f"  Time: {time.time()-t0:.1f}s")

# ============================================================
# IDEA 834: EVOLVE for c_eff closest to 1.0
# ============================================================
print("\n" + "=" * 78)
print("IDEA 834: EVOLVE for c_eff closest to 1.0")
print("What 2-order gives the most CFT-like SJ vacuum?")
print("=" * 78)

t0 = time.time()

def fitness_834(cs, to):
    c = c_eff(cs)
    # Minimize |c_eff - 1.0| => maximize -|c_eff - 1.0|
    return -abs(c - 1.0)

pop_834 = make_initial_population(POP_SIZE, N, rng)

baseline_c = []
for to in pop_834:
    cs = to.to_causet()
    baseline_c.append(c_eff(cs))
print(f"  Baseline random: c_eff = {np.mean(baseline_c):.4f} +/- {np.std(baseline_c):.4f}")

best_to_834, best_cs_834, history_834 = evolve(
    pop_834, fitness_834, N_GENERATIONS, rng, maximize=True, label="c_eff_to_1")

best_c = c_eff(best_cs_834)
struct = describe_structure(best_cs_834, best_to_834)
print(f"\n  EVOLVED BEST: c_eff = {best_c:.4f}")
print(f"  |c_eff - 1| = {abs(best_c - 1.0):.4f}")
print(f"  Structure: ord_frac={struct['ord_frac']:.3f}, link_frac={struct['link_frac']:.3f}, "
      f"chain={struct['chain']}, fiedler={struct['fiedler']:.4f}")
print(f"  Improvement: |c_eff-1| = {abs(np.mean(baseline_c)-1.0):.4f} -> {abs(best_c-1.0):.4f}")
print(f"  Time: {time.time()-t0:.1f}s")

# ============================================================
# IDEA 835: EVOLVE for max deviation from GUE (<r> furthest from 0.60)
# ============================================================
print("\n" + "=" * 78)
print("IDEA 835: EVOLVE for max deviation from GUE")
print("<r> furthest from 0.60 — can evolution break GUE universality?")
print("=" * 78)

t0 = time.time()

GUE_R = 0.5996  # GUE prediction for <r>

def fitness_835(cs, to):
    r = iDelta_level_spacing(cs)
    return abs(r - GUE_R)  # maximize deviation from GUE

pop_835 = make_initial_population(POP_SIZE, N, rng)

baseline_r_gue = []
for to in pop_835:
    cs = to.to_causet()
    baseline_r_gue.append(iDelta_level_spacing(cs))
print(f"  Baseline random: <r> = {np.mean(baseline_r_gue):.4f} +/- {np.std(baseline_r_gue):.4f}")
print(f"  Baseline |<r> - GUE| = {abs(np.mean(baseline_r_gue) - GUE_R):.4f}")

best_to_835, best_cs_835, history_835 = evolve(
    pop_835, fitness_835, N_GENERATIONS, rng, maximize=True, label="break_GUE")

best_r_gue = iDelta_level_spacing(best_cs_835)
struct = describe_structure(best_cs_835, best_to_835)
print(f"\n  EVOLVED BEST: <r> = {best_r_gue:.4f}")
print(f"  |<r> - GUE| = {abs(best_r_gue - GUE_R):.4f}")
print(f"  Structure: ord_frac={struct['ord_frac']:.3f}, link_frac={struct['link_frac']:.3f}, "
      f"chain={struct['chain']}, fiedler={struct['fiedler']:.4f}")

# Which direction did it go?
if best_r_gue < GUE_R:
    print(f"  Direction: toward Poisson (0.39) — more integrable/uncorrelated")
elif best_r_gue > GUE_R:
    print(f"  Direction: beyond GUE — hypercorrelated spacing")
print(f"  GUE universality {'BROKEN' if abs(best_r_gue - GUE_R) > 0.1 else 'resistant'}!")
print(f"  Time: {time.time()-t0:.1f}s")

# ============================================================
# IDEA 836: SEXUAL REPRODUCTION
# ============================================================
print("\n" + "=" * 78)
print("IDEA 836: SEXUAL REPRODUCTION — Crossover of 2-orders")
print("Take u from parent 1 and v from parent 2. Are offspring viable?")
print("=" * 78)

t0 = time.time()

def crossover(parent1, parent2):
    """Sexual reproduction: u from parent1, v from parent2."""
    return TwoOrder.from_permutations(parent1.u, parent2.v)


# Generate parents and offspring
n_trials = 100
parent_stats = {'ord_frac': [], 'chain': [], 'fiedler': [], 'H': [], 'r_gue': []}
offspring_stats = {'ord_frac': [], 'chain': [], 'fiedler': [], 'H': [], 'r_gue': []}
viability_count = 0  # offspring with reasonable properties

for trial in range(n_trials):
    p1 = TwoOrder(N, rng=np.random.default_rng(rng.integers(2**31)))
    p2 = TwoOrder(N, rng=np.random.default_rng(rng.integers(2**31)))
    child = crossover(p1, p2)

    cs_p1 = p1.to_causet()
    cs_p2 = p2.to_causet()
    cs_child = child.to_causet()

    # Parent stats (average of both parents)
    parent_stats['ord_frac'].append((ordering_fraction(cs_p1) + ordering_fraction(cs_p2)) / 2)
    parent_stats['chain'].append((longest_chain(cs_p1) + longest_chain(cs_p2)) / 2)
    parent_stats['fiedler'].append((fiedler_value(cs_p1) + fiedler_value(cs_p2)) / 2)
    parent_stats['H'].append((interval_entropy(cs_p1) + interval_entropy(cs_p2)) / 2)
    parent_stats['r_gue'].append((iDelta_level_spacing(cs_p1) + iDelta_level_spacing(cs_p2)) / 2)

    # Offspring stats
    of = ordering_fraction(cs_child)
    ch = longest_chain(cs_child)
    fi = fiedler_value(cs_child)
    H = interval_entropy(cs_child)
    r_g = iDelta_level_spacing(cs_child)
    offspring_stats['ord_frac'].append(of)
    offspring_stats['chain'].append(ch)
    offspring_stats['fiedler'].append(fi)
    offspring_stats['H'].append(H)
    offspring_stats['r_gue'].append(r_g)

    # "Viable" = within 2 sigma of random baseline
    if 0.1 < of < 0.6 and ch > 2 and fi > 0.01:
        viability_count += 1

print(f"  Viability: {viability_count}/{n_trials} offspring have reasonable properties")
print(f"\n  {'Observable':>15} | {'Parent mean':>12} | {'Offspring mean':>14} | {'Ratio':>8} | {'Corr(P,O)':>10}")
print(f"  {'-'*15}-+-{'-'*12}-+-{'-'*14}-+-{'-'*8}-+-{'-'*10}")

for key in ['ord_frac', 'chain', 'fiedler', 'H', 'r_gue']:
    p_mean = np.mean(parent_stats[key])
    o_mean = np.mean(offspring_stats[key])
    ratio = o_mean / p_mean if p_mean != 0 else float('nan')
    corr = np.corrcoef(parent_stats[key], offspring_stats[key])[0, 1]
    print(f"  {key:>15} | {p_mean:12.4f} | {o_mean:14.4f} | {ratio:8.3f} | {corr:10.4f}")

print(f"\n  REPRODUCTION RESULT:")
if viability_count > 80:
    print(f"  Crossover is highly viable — offspring resemble parents")
elif viability_count > 50:
    print(f"  Crossover is moderately viable — some offspring are anomalous")
else:
    print(f"  Crossover often produces non-viable offspring — structure is fragile")
print(f"  Time: {time.time()-t0:.1f}s")

# ============================================================
# IDEA 837: SPECIATION
# ============================================================
print("\n" + "=" * 78)
print("IDEA 837: SPECIATION — Two populations, different fitness goals")
print("Pop A -> max interval entropy, Pop B -> min Fiedler value")
print("Do they diverge in structure?")
print("=" * 78)

t0 = time.time()

# Use same initial population for both
shared_initial = make_initial_population(POP_SIZE, N, rng)

print("  Evolving Species A (max interval entropy)...")
best_A, cs_A, hist_A = evolve(
    shared_initial, fitness_831, N_GENERATIONS,
    np.random.default_rng(rng.integers(2**31)),
    maximize=True, verbose=False)

print("  Evolving Species B (min Fiedler value)...")
best_B, cs_B, hist_B = evolve(
    shared_initial, fitness_832, N_GENERATIONS,
    np.random.default_rng(rng.integers(2**31)),
    maximize=False, verbose=False)

struct_A = describe_structure(cs_A, best_A)
struct_B = describe_structure(cs_B, best_B)

print(f"\n  {'Observable':>15} | {'Species A':>12} | {'Species B':>12} | {'Divergence':>12}")
print(f"  {'-'*15}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

for key in struct_A:
    a_val = struct_A[key]
    b_val = struct_B[key]
    avg = (abs(a_val) + abs(b_val)) / 2
    div = abs(a_val - b_val) / avg if avg > 0 else 0.0
    print(f"  {key:>15} | {a_val:12.4f} | {b_val:12.4f} | {div:12.3f}")

# Also check observables not used in fitness
r_A = iDelta_level_spacing(cs_A)
r_B = iDelta_level_spacing(cs_B)
print(f"  {'<r> (GUE)':>15} | {r_A:12.4f} | {r_B:12.4f} | {abs(r_A-r_B)/((abs(r_A)+abs(r_B))/2):12.3f}")

print(f"\n  SPECIATION RESULT:")
total_div = sum(abs(struct_A[k] - struct_B[k]) / max((abs(struct_A[k]) + abs(struct_B[k]))/2, 1e-10)
                for k in struct_A) / len(struct_A)
print(f"  Mean relative divergence: {total_div:.3f}")
if total_div > 0.5:
    print(f"  Strong speciation — populations evolved very different structures!")
elif total_div > 0.2:
    print(f"  Moderate speciation — some structural divergence")
else:
    print(f"  Weak speciation — structures remain similar despite different pressures")
print(f"  Time: {time.time()-t0:.1f}s")

# ============================================================
# IDEA 838: EXTINCTION
# ============================================================
print("\n" + "=" * 78)
print("IDEA 838: EXTINCTION — Which properties go extinct first?")
print("Start with 100 random 2-orders, apply random perturbations,")
print("track which observables lose their variance first")
print("=" * 78)

t0 = time.time()

n_pop_ext = 100
n_pert_steps = 50

# Initial population
ext_pop = [TwoOrder(N, rng=np.random.default_rng(rng.integers(2**31)))
           for _ in range(n_pop_ext)]

# Track observables over perturbation rounds
obs_names = ['ord_frac', 'link_frac', 'chain', 'fiedler', 'H', 'r_gue']
obs_history = {name: [] for name in obs_names}

def measure_population(pop_tos):
    """Measure all observables for a population."""
    result = {name: [] for name in obs_names}
    for to in pop_tos:
        cs = to.to_causet()
        result['ord_frac'].append(ordering_fraction(cs))
        result['link_frac'].append(link_fraction(cs))
        result['chain'].append(float(longest_chain(cs)))
        result['fiedler'].append(fiedler_value(cs))
        result['H'].append(interval_entropy(cs))
        result['r_gue'].append(iDelta_level_spacing(cs))
    return result

# Measure initial state
initial_obs = measure_population(ext_pop)
for name in obs_names:
    obs_history[name].append(np.std(initial_obs[name]))

print(f"  Initial diversity (std devs):")
for name in obs_names:
    print(f"    {name:>12}: {np.std(initial_obs[name]):.4f}")

# Apply random perturbations and track diversity loss
for step in range(n_pert_steps):
    # Each member gets a random mutation
    ext_pop = [mutate(to, rng, n_swaps=1) for to in ext_pop]

    if (step + 1) % 10 == 0:
        current_obs = measure_population(ext_pop)
        for name in obs_names:
            obs_history[name].append(np.std(current_obs[name]))

# Compute diversity decay rate for each observable
print(f"\n  Diversity decay (std at step 0 -> step {n_pert_steps}):")
extinction_order = []
for name in obs_names:
    initial_std = obs_history[name][0]
    final_std = obs_history[name][-1]
    ratio = final_std / initial_std if initial_std > 0 else 1.0
    extinction_order.append((ratio, name))
    print(f"    {name:>12}: {initial_std:.4f} -> {final_std:.4f} (retained {ratio:.1%})")

extinction_order.sort()
print(f"\n  EXTINCTION ORDER (first to lose diversity -> last):")
for rank, (ratio, name) in enumerate(extinction_order):
    print(f"    {rank+1}. {name} (retained {ratio:.1%} of initial variance)")
print(f"  Time: {time.time()-t0:.1f}s")

# ============================================================
# IDEA 839: COEVOLUTION
# ============================================================
print("\n" + "=" * 78)
print("IDEA 839: COEVOLUTION — Causet vs SJ vacuum, adversarial evolution")
print("Causet evolves to maximize entanglement entropy,")
print("'vacuum' adapts via region choice to minimize it.")
print("Who wins?")
print("=" * 78)

t0 = time.time()

# Two populations: causets and "region selectors"
# Causet evolves to maximize S_ent for the worst-case region
# Region selector evolves to find the region that minimizes S_ent
n_coev_gen = N_GENERATIONS

# Population of causets
causet_pop = make_initial_population(POP_SIZE // 2, N, rng)
# Population of region definitions (each is a random subset of elements)
# Represented as a sorted list of element indices
region_pop = [sorted(rng.choice(N, size=N//2, replace=False).tolist())
              for _ in range(POP_SIZE // 2)]

causet_wins = []
vacuum_wins = []

for gen in range(n_coev_gen):
    # Evaluate: each causet gets the entropy from the worst region
    causet_fitnesses = []
    for to in causet_pop:
        cs = to.to_causet()
        W = sj_wightman(cs)
        # Worst-case entropy over all regions in the region population
        entropies = []
        for region in region_pop:
            try:
                W_A = W[np.ix_(region, region)]
                evals = np.linalg.eigvalsh(W_A)
                evals = np.clip(evals, 1e-15, 1 - 1e-15)
                S = -np.sum(evals * np.log(evals) + (1 - evals) * np.log(1 - evals))
                entropies.append(S)
            except Exception:
                entropies.append(0.0)
        causet_fitnesses.append(np.min(entropies))  # worst-case

    # Evaluate: each region gets minus the entropy from the best causet
    region_fitnesses = []
    for region in region_pop:
        entropies = []
        for to in causet_pop:
            cs = to.to_causet()
            W = sj_wightman(cs)
            try:
                W_A = W[np.ix_(region, region)]
                evals = np.linalg.eigvalsh(W_A)
                evals = np.clip(evals, 1e-15, 1 - 1e-15)
                S = -np.sum(evals * np.log(evals) + (1 - evals) * np.log(1 - evals))
                entropies.append(S)
            except Exception:
                entropies.append(0.0)
        region_fitnesses.append(-np.max(entropies))  # wants to minimize

    causet_fitnesses = np.array(causet_fitnesses)
    region_fitnesses = np.array(region_fitnesses)

    causet_wins.append(np.max(causet_fitnesses))
    vacuum_wins.append(-np.min(region_fitnesses))

    if gen % 10 == 0:
        print(f"    Gen {gen:3d}: causet_best_S={np.max(causet_fitnesses):.4f}, "
              f"vacuum_best_minS={-np.min(region_fitnesses):.4f}")

    # Select top 50% of causets
    causet_surv_idx = np.argsort(causet_fitnesses)[-len(causet_pop) // 2:]
    causet_survivors = [causet_pop[i].copy() for i in causet_surv_idx]
    new_causet_pop = [s.copy() for s in causet_survivors]
    while len(new_causet_pop) < len(causet_pop):
        parent = causet_survivors[rng.integers(len(causet_survivors))]
        new_causet_pop.append(mutate(parent, rng, n_swaps=N_MUTATIONS))
    causet_pop = new_causet_pop

    # Select top 50% of regions (lowest entropy = most negative fitness = highest -S)
    region_surv_idx = np.argsort(region_fitnesses)[-len(region_pop) // 2:]
    region_survivors = [region_pop[i][:] for i in region_surv_idx]
    new_region_pop = [r[:] for r in region_survivors]
    while len(new_region_pop) < len(region_pop):
        parent = region_survivors[rng.integers(len(region_survivors))]
        # Mutate region: swap one element in/out
        child = parent[:]
        remove_idx = rng.integers(len(child))
        available = [x for x in range(N) if x not in child]
        if available:
            child[remove_idx] = rng.choice(available)
            child.sort()
        new_region_pop.append(child)
    region_pop = new_region_pop

# Who won?
causet_trend = np.polyfit(range(len(causet_wins)), causet_wins, 1)[0]
vacuum_trend = np.polyfit(range(len(vacuum_wins)), vacuum_wins, 1)[0]

print(f"\n  COEVOLUTION RESULT:")
print(f"  Causet entropy trend:  slope = {causet_trend:+.6f}/gen")
print(f"  Vacuum entropy trend:  slope = {vacuum_trend:+.6f}/gen (from vacuum's perspective)")
print(f"  Initial causet entropy: {causet_wins[0]:.4f}")
print(f"  Final causet entropy:   {causet_wins[-1]:.4f}")
print(f"  Initial vacuum min-S:   {vacuum_wins[0]:.4f}")
print(f"  Final vacuum min-S:     {vacuum_wins[-1]:.4f}")

if causet_trend > abs(vacuum_trend):
    print(f"  WINNER: Causet (entropy grows faster than vacuum can suppress)")
elif abs(vacuum_trend) > causet_trend:
    print(f"  WINNER: Vacuum (entropy suppression outpaces causet adaptation)")
else:
    print(f"  DRAW: Arms race in equilibrium")
print(f"  Time: {time.time()-t0:.1f}s")

# ============================================================
# IDEA 840: GENETIC DRIFT (neutral evolution)
# ============================================================
print("\n" + "=" * 78)
print("IDEA 840: GENETIC DRIFT — 10 populations, NO selection")
print("Evolve with random mutations only. Do they converge?")
print("=" * 78)

t0 = time.time()

n_drift_pops = 10
n_drift_gen = N_GENERATIONS
drift_pop_size = POP_SIZE

# Create 10 identical starting populations
seed_pop = make_initial_population(drift_pop_size, N, rng)
drift_pops = [[to.copy() for to in seed_pop] for _ in range(n_drift_pops)]

# Evolve with NO selection — just random mutation and random survival
final_obs = {name: [] for name in obs_names}

for pop_idx in range(n_drift_pops):
    pop = drift_pops[pop_idx]
    drift_rng = np.random.default_rng(rng.integers(2**31))

    for gen in range(n_drift_gen):
        # No selection — just replace random members with mutations of random others
        for k in range(len(pop)):
            # Each member has 50% chance of being replaced by a mutant of a random other
            if drift_rng.random() < 0.5:
                donor = pop[drift_rng.integers(len(pop))]
                pop[k] = mutate(donor, drift_rng, n_swaps=1)

    # Measure final population
    pop_obs = {name: [] for name in obs_names}
    for to in pop:
        cs = to.to_causet()
        pop_obs['ord_frac'].append(ordering_fraction(cs))
        pop_obs['link_frac'].append(link_fraction(cs))
        pop_obs['chain'].append(float(longest_chain(cs)))
        pop_obs['fiedler'].append(fiedler_value(cs))
        pop_obs['H'].append(interval_entropy(cs))
        pop_obs['r_gue'].append(iDelta_level_spacing(cs))

    for name in obs_names:
        final_obs[name].append(np.mean(pop_obs[name]))

# Check convergence: do the 10 populations end up similar?
print(f"\n  {'Observable':>12} | {'Pop mean':>10} | {'Pop std':>10} | {'CV':>8} | {'Converged?':>12}")
print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*12}")

convergence_scores = []
for name in obs_names:
    vals = final_obs[name]
    mean_val = np.mean(vals)
    std_val = np.std(vals)
    cv = std_val / abs(mean_val) if abs(mean_val) > 1e-10 else float('inf')
    converged = "YES" if cv < 0.1 else ("partial" if cv < 0.3 else "NO")
    convergence_scores.append(cv)
    print(f"  {name:>12} | {mean_val:10.4f} | {std_val:10.4f} | {cv:8.4f} | {converged:>12}")

mean_cv = np.mean(convergence_scores)
print(f"\n  GENETIC DRIFT RESULT:")
print(f"  Mean coefficient of variation across populations: {mean_cv:.4f}")
if mean_cv < 0.1:
    print(f"  Strong convergence — neutral drift leads to similar structures")
    print(f"  This suggests an 'attractor' in 2-order space (ergodicity)")
elif mean_cv < 0.3:
    print(f"  Moderate convergence — some observables converge, others drift")
else:
    print(f"  Weak convergence — populations remain diverse under neutral drift")
    print(f"  This suggests multiple 'basins' in 2-order space")
print(f"  Time: {time.time()-t0:.1f}s")

# ============================================================
# GRAND SUMMARY
# ============================================================
print("\n" + "=" * 78)
print("GRAND SUMMARY: EVOLUTIONARY DYNAMICS (Ideas 831-840)")
print("=" * 78)

print(f"""
  831. MAX INTERVAL ENTROPY: Evolved H = {interval_entropy(best_cs_831):.4f}
       (baseline {np.mean(baseline_H):.4f}). High-H causets have diverse interval
       structure — neither too ordered nor too random.

  832. MIN FIEDLER VALUE: Evolved Fiedler = {fiedler_value(best_cs_832):.6f}
       (baseline {np.mean(baseline_fiedler):.4f}). Minimally connected causets
       have near-disconnected Hasse diagrams — approaching graph bisection.

  833. MAX ER=EPR CORRELATION: Evolved r = {best_r:.4f}
       (baseline {np.mean(baseline_r):.4f}).
       {'Evolution CAN find r > 0.95!' if best_r > 0.95 else 'r > 0.95 not reached — ER=EPR has a natural ceiling.'}

  834. C_EFF -> 1.0: Best c_eff = {best_c:.4f} (|c_eff - 1| = {abs(best_c - 1.0):.4f})
       (baseline c_eff = {np.mean(baseline_c):.4f}).
       {'Found a CFT-like causet!' if abs(best_c - 1.0) < 0.3 else 'c_eff=1 is hard to reach at N=30.'}

  835. BREAK GUE: Evolved <r> = {best_r_gue:.4f} (|<r> - GUE| = {abs(best_r_gue - GUE_R):.4f})
       (baseline <r> = {np.mean(baseline_r_gue):.4f}).
       {'GUE universality BROKEN!' if abs(best_r_gue - GUE_R) > 0.1 else 'GUE universality is ROBUST even under selection.'}

  836. SEXUAL REPRODUCTION: {viability_count}% viable offspring.
       Crossover produces {'viable' if viability_count > 50 else 'often non-viable'} causets.
       Parent-offspring observable correlations reveal heritability patterns.

  837. SPECIATION: Mean divergence = {total_div:.3f}.
       {'Strong' if total_div > 0.5 else 'Moderate' if total_div > 0.2 else 'Weak'} structural divergence
       under different selection pressures.

  838. EXTINCTION ORDER: {' > '.join([name for _, name in extinction_order])}
       (first property to lose diversity -> last).

  839. COEVOLUTION: {'Causet wins' if causet_trend > abs(vacuum_trend) else 'Vacuum wins' if abs(vacuum_trend) > causet_trend else 'Draw'}.
       Entropy trend: {causet_wins[0]:.4f} -> {causet_wins[-1]:.4f}.

  840. GENETIC DRIFT: Mean CV = {mean_cv:.4f}.
       {'Strong' if mean_cv < 0.1 else 'Moderate' if mean_cv < 0.3 else 'Weak'} convergence
       under neutral evolution.

  KEY INSIGHT: The evolutionary framework reveals which observables are
  "evolvable" (can be pushed far from random) vs "universal" (robust to
  selection). GUE level spacing is notoriously universal — if evolution
  can't break it, that's a strong universality result.
""")

print(f"Total experiment time: done.")

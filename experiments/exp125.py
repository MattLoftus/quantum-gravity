"""
Experiment 125: PERTURBATION THEORY — Start from known exact results, perturb them.

METHODOLOGY: Take a known exact result in causal set theory. Apply a small
perturbation (add/remove one relation, one element, one constraint). Measure
how the answer shifts. This is the physicist's most reliable tool.

761. E[f] = 1/2 for random 2-orders. Add ONE extra causal relation. What's E[f|one extra]?
762. E[links] = (N+1)H_N - 2N. Remove one element. Does E[links|N-1] follow formula at N-1?
763. Kronecker CDT theorem. Add ONE random relation within a time slice. How does n_pos change?
764. GUE spacing <r>=0.60. Impose one specific relation on causal matrix. How does <r> shift?
765. Master interval formula P[k|m]=2(m-k)/[m(m+1)]. First correction for d=3?
766. Total order (chain, <r>->1). Add random spacelike pairs. Plot <r> vs #spacelike.
767. S(N/2) = c/3*ln(N). Remove one element from center. How does S change?
768. Fiedler value lambda_2. Remove highest-degree element. How does lambda_2 change?
769. Exact Z(beta) at N=4. Add N=5 corrections. Does perturbation series converge?
770. Antichain ~ 2*sqrt(N). Force one spacelike pair to be causal. How does antichain change?
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats, special
from scipy.linalg import eigh, eigvalsh
from scipy.optimize import curve_fit
import time

from causal_sets.fast_core import FastCausalSet, sprinkle_fast
from causal_sets.two_orders import TwoOrder, swap_move
from causal_sets.d_orders import DOrder
from causal_sets.bd_action import count_intervals_by_size, count_links
from causal_sets.sj_vacuum import pauli_jordan_function, sj_wightman_function
from causal_sets.two_orders_v2 import bd_action_corrected

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


def ordering_fraction(cs):
    """Fraction of pairs that are causally related."""
    if cs.n < 2:
        return 0.0
    return cs.num_relations() / (cs.n * (cs.n - 1) / 2)


def longest_antichain(cs):
    """Find the longest antichain via greedy clique on incomparability graph."""
    N = cs.n
    related = cs.order | cs.order.T
    np.fill_diagonal(related, True)
    incomp = ~related
    np.fill_diagonal(incomp, False)

    if N <= 30:
        best = 0
        for _ in range(min(50, N)):
            start = rng.integers(N)
            clique = [start]
            candidates = list(np.where(incomp[start])[0])
            rng.shuffle(candidates)
            for c in candidates:
                if all(incomp[c, x] for x in clique):
                    clique.append(c)
            best = max(best, len(clique))
        return best
    else:
        best = 0
        for _ in range(200):
            start = rng.integers(N)
            clique = [start]
            perm = rng.permutation(N)
            for c in perm:
                if c == start:
                    continue
                if all(incomp[c, x] for x in clique):
                    clique.append(c)
            best = max(best, len(clique))
        return best


def sj_entropy(cs, region_indices):
    """Compute SJ entanglement entropy for a subregion."""
    W = sj_wightman_function(cs)
    W_A = W[np.ix_(region_indices, region_indices)]
    evals = np.linalg.eigvalsh(W_A)
    evals = np.clip(evals, 1e-15, 1 - 1e-15)
    S = -np.sum(evals * np.log(evals + 1e-30) + (1 - evals) * np.log(1 - evals + 1e-30))
    return S


def spectral_gap_ratio(eigenvalues):
    """Compute mean gap ratio <r> for level spacing statistics."""
    evals = np.sort(eigenvalues)
    gaps = np.diff(evals)
    gaps = gaps[gaps > 1e-14]
    if len(gaps) < 2:
        return 0.0
    ratios = []
    for i in range(len(gaps) - 1):
        r = min(gaps[i], gaps[i + 1]) / max(gaps[i], gaps[i + 1])
        ratios.append(r)
    return np.mean(ratios)


print("=" * 80)
print("EXPERIMENT 125: PERTURBATION THEORY")
print("Start from known exact results, apply small perturbations")
print("=" * 80)
sys.stdout.flush()


# ============================================================
# IDEA 761: E[f]=1/2 for random 2-orders. Add ONE extra relation.
# ============================================================
print("\n" + "=" * 80)
print("IDEA 761: Perturbing E[f]=1/2 by adding ONE causal relation")
print("=" * 80)
print("""
BACKGROUND: For a random 2-order on N elements, the ordering fraction f
(fraction of pairs that are causally related) has E[f] = 1/2 exactly
(by symmetry of the two independent permutations).

PERTURBATION: Take a random 2-order. Find a SPACELIKE pair (i,j) and
force i < j (add the relation + transitive closure). Measure the new f.
The shift delta_f = f_new - f_old tells us how "rigid" the ordering fraction is.

QUESTION: What is E[delta_f] as a function of N? Does adding one relation
cascade through transitivity?
""")
sys.stdout.flush()

t0 = time.time()

print(f"  {'N':>4} {'E[f] before':>12} {'E[f] after':>12} {'E[delta_f]':>12} {'std[delta_f]':>12} "
      f"{'E[delta_rels]':>14} {'cascade ratio':>14}")
print("  " + "-" * 90)

for N in [10, 20, 30, 50, 80, 100]:
    n_trials = 300 if N <= 50 else 100
    f_before_list = []
    f_after_list = []
    delta_f_list = []
    delta_rels_list = []

    for trial in range(n_trials):
        r = np.random.default_rng(trial * 1000 + N)
        cs, to = random_2order(N, rng_local=r)

        f_before = ordering_fraction(cs)
        n_rels_before = cs.num_relations()

        # Find a spacelike pair
        spacelike_pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                if not cs.order[i, j] and not cs.order[j, i]:
                    spacelike_pairs.append((i, j))

        if len(spacelike_pairs) == 0:
            continue

        idx = r.integers(len(spacelike_pairs))
        a, b = spacelike_pairs[idx]

        # Add a < b and take transitive closure
        new_order = cs.order.copy()
        new_order[a, b] = True

        changed = True
        while changed:
            old = new_order.copy()
            new_order = new_order | (new_order @ new_order).astype(bool)
            np.fill_diagonal(new_order, False)
            changed = np.any(new_order != old)

        cs_new = FastCausalSet(N)
        cs_new.order = new_order

        f_after = ordering_fraction(cs_new)
        n_rels_after = cs_new.num_relations()

        f_before_list.append(f_before)
        f_after_list.append(f_after)
        delta_f_list.append(f_after - f_before)
        delta_rels_list.append(n_rels_after - n_rels_before)

    max_pairs = N * (N - 1) / 2
    mean_delta_rels = np.mean(delta_rels_list)

    print(f"  {N:>4} {np.mean(f_before_list):>12.6f} {np.mean(f_after_list):>12.6f} "
          f"{np.mean(delta_f_list):>12.6f} {np.std(delta_f_list):>12.6f} "
          f"{mean_delta_rels:>14.2f} {mean_delta_rels:>14.2f}")

print(f"\n  THEORY: Adding one relation to a 2-order gives delta_f = 1/C(N,2) + cascade.")
print(f"  The cascade comes from transitive closure: if we force a<b, then")
print(f"  all x<a also get x<b, and all b<y also get a<y, etc.")
print(f"  Minimal delta_f (no cascade) = 1/C(N,2) = 2/[N(N-1)]:")
for N in [10, 20, 50, 100]:
    print(f"    N={N}: minimal delta_f = {2 / (N * (N - 1)):.6f}")

dt = time.time() - t0
print(f"\n  [Idea 761 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# IDEA 762: E[links] formula under element removal
# ============================================================
print("\n" + "=" * 80)
print("IDEA 762: E[links] = (N+1)H_N - 2N under element removal")
print("=" * 80)
print("""
BACKGROUND: For random 2-orders, E[links] = (N+1)*H_N - 2N where H_N
is the N-th harmonic number. This is an EXACT result.

PERTURBATION: Remove one element (uniformly at random) from a random
2-order on N elements. The result is a poset on N-1 elements.
Is it still a random 2-order? If so, E[links|N-1] = N*H_{N-1} - 2(N-1).
If NOT (element removal breaks the 2-order property), the measured value
will differ from the formula.

QUESTION: Does element removal preserve the 2-order distribution?
""")
sys.stdout.flush()

t0 = time.time()

def harmonic(n):
    return sum(1.0 / k for k in range(1, n + 1))

print(f"  {'N':>4} {'E[links] formula(N)':>20} {'E[links] measured(N)':>22} "
      f"{'E[links] after removal':>22} {'Formula at N-1':>18} {'Ratio':>8}")
print("  " + "-" * 100)

for N in [10, 15, 20, 30, 50]:
    n_trials = 500 if N <= 30 else 200
    formula_N = (N + 1) * harmonic(N) - 2 * N
    formula_Nm1 = N * harmonic(N - 1) - 2 * (N - 1)

    links_original = []
    links_after_removal = []

    for trial in range(n_trials):
        r = np.random.default_rng(trial * 2000 + N * 7)
        cs, to = random_2order(N, rng_local=r)

        n_links_orig = count_links(cs)
        links_original.append(n_links_orig)

        elem = r.integers(N)
        keep = [i for i in range(N) if i != elem]
        new_order = cs.order[np.ix_(keep, keep)]

        cs_reduced = FastCausalSet(N - 1)
        cs_reduced.order = new_order

        n_links_reduced = count_links(cs_reduced)
        links_after_removal.append(n_links_reduced)

    mean_orig = np.mean(links_original)
    mean_reduced = np.mean(links_after_removal)
    ratio = mean_reduced / formula_Nm1 if formula_Nm1 > 0 else float('nan')

    print(f"  {N:>4} {formula_N:>20.4f} {mean_orig:>22.4f} "
          f"{mean_reduced:>22.4f} {formula_Nm1:>18.4f} {ratio:>8.4f}")

print(f"\n  INTERPRETATION:")
print(f"  If ratio ~ 1.000, element removal preserves the 2-order distribution.")
print(f"  If ratio != 1, the induced subposet on N-1 elements is NOT a uniform")
print(f"  random 2-order (the marginal distribution is biased).")

dt = time.time() - t0
print(f"\n  [Idea 762 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# IDEA 763: Kronecker CDT + one random within-slice relation
# ============================================================
print("\n" + "=" * 80)
print("IDEA 763: Perturbing CDT Kronecker structure with one intra-slice relation")
print("=" * 80)
print("""
BACKGROUND: For uniform CDT with T time slices of s spatial elements,
the antisymmetric causal matrix M = C^T - C satisfies M = A_T (x) J_s
EXACTLY. The number of positive SJ modes (n_pos) is exactly (T-1)/2
(for odd T).

PERTURBATION: Add ONE causal relation WITHIN a single time slice
(breaking the "no spatial relations" rule of CDT). How does n_pos change?
Does the Kronecker structure break catastrophically or perturbatively?
""")
sys.stdout.flush()

t0 = time.time()

print(f"  {'T*s':>8} {'N':>4} {'n_pos (CDT)':>12} {'n_pos (perturbed)':>18} "
      f"{'delta_n_pos':>12} {'||dM||/||M||':>14}")
print("  " + "-" * 75)

for T, s in [(5, 4), (7, 3), (9, 3), (7, 5), (11, 3)]:
    N = T * s

    cs_cdt = FastCausalSet(N)
    for t1 in range(T):
        for t2 in range(t1 + 1, T):
            for s1 in range(s):
                for s2 in range(s):
                    cs_cdt.order[t1 * s + s1, t2 * s + s2] = True

    iDelta_cdt = pauli_jordan_function(cs_cdt)
    evals_cdt = np.linalg.eigvalsh(1j * iDelta_cdt).real
    n_pos_cdt = np.sum(evals_cdt > 1e-10)
    M_cdt = cs_cdt.order.astype(float).T - cs_cdt.order.astype(float)
    M_norm = np.linalg.norm(M_cdt, 'fro')

    n_pos_perturbed_list = []
    delta_M_list = []

    n_perturb_trials = 20
    for trial in range(n_perturb_trials):
        r = np.random.default_rng(trial * 500 + T * 13 + s * 7)

        t_slice = r.integers(T)
        s1_idx, s2_idx = r.choice(s, size=2, replace=False)
        i = t_slice * s + s1_idx
        j = t_slice * s + s2_idx
        if i > j:
            i, j = j, i

        cs_pert = FastCausalSet(N)
        cs_pert.order = cs_cdt.order.copy()
        cs_pert.order[i, j] = True

        changed = True
        while changed:
            old = cs_pert.order.copy()
            cs_pert.order = cs_pert.order | (cs_pert.order.astype(np.int8) @ cs_pert.order.astype(np.int8)).astype(bool)
            np.fill_diagonal(cs_pert.order, False)
            changed = np.any(cs_pert.order != old)

        M_pert = cs_pert.order.astype(float).T - cs_pert.order.astype(float)
        delta_M = np.linalg.norm(M_pert - M_cdt, 'fro') / M_norm

        iDelta_pert = pauli_jordan_function(cs_pert)
        evals_pert = np.linalg.eigvalsh(1j * iDelta_pert).real
        n_pos_pert = np.sum(evals_pert > 1e-10)

        n_pos_perturbed_list.append(n_pos_pert)
        delta_M_list.append(delta_M)

    mean_n_pos_pert = np.mean(n_pos_perturbed_list)
    mean_delta_M = np.mean(delta_M_list)

    print(f"  {T}x{s:>3} {N:>4} {n_pos_cdt:>12} {mean_n_pos_pert:>18.2f} "
          f"{mean_n_pos_pert - n_pos_cdt:>12.2f} {mean_delta_M:>14.6f}")

print(f"\n  INTERPRETATION: If delta_n_pos ~ 0, the SJ vacuum is STABLE under")
print(f"  single intra-slice perturbations. If delta_n_pos ~ O(1), one relation")
print(f"  can fundamentally change the quantum field theory on the spacetime.")

dt = time.time() - t0
print(f"\n  [Idea 763 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# IDEA 764: GUE <r>=0.60, perturb causal matrix by one constraint
# ============================================================
print("\n" + "=" * 80)
print("IDEA 764: Perturbing GUE universality by imposing one causal constraint")
print("=" * 80)
print("""
BACKGROUND: The Pauli-Jordan matrix iDelta of a random 2-order has level
spacing statistics in the GUE universality class (<r> ~ 0.5996).

PERTURBATION: Start from a random 2-order. Fix one specific relation
(e.g., force element 0 < element N-1 regardless of the permutation).
How does <r> shift? Does one constraint break GUE universality?

ALSO: What if we impose the constraint element 0 || element N-1
(force spacelike)? This constrains the causal matrix differently.
""")
sys.stdout.flush()

t0 = time.time()

GUE_R = 0.5996
POISSON_R = 0.3863

print(f"  Reference: GUE <r> = {GUE_R}, Poisson <r> = {POISSON_R}")
print()
print(f"  {'N':>4} {'<r> unconstrained':>18} {'<r> force causal':>18} "
      f"{'<r> force spacelike':>20} {'dr (causal)':>12} {'dr (spacelike)':>16}")
print("  " + "-" * 95)

for N in [20, 30, 50, 80]:
    n_trials = 200 if N <= 50 else 100

    r_unconstrained = []
    r_force_causal = []
    r_force_spacelike = []

    for trial in range(n_trials):
        rl = np.random.default_rng(trial * 3000 + N * 11)

        # Unconstrained
        cs, to = random_2order(N, rng_local=rl)
        iDelta = pauli_jordan_function(cs)
        evals = np.linalg.eigvalsh(1j * iDelta).real
        r_unconstrained.append(spectral_gap_ratio(evals))

        # Force element 0 < element N-1 (causal)
        to2 = TwoOrder(N, rng=np.random.default_rng(trial * 3000 + N * 11 + 1))
        if to2.u[0] > to2.u[N - 1]:
            to2.u[0], to2.u[N - 1] = to2.u[N - 1], to2.u[0]
        if to2.v[0] > to2.v[N - 1]:
            to2.v[0], to2.v[N - 1] = to2.v[N - 1], to2.v[0]
        cs2 = to2.to_causet()
        iDelta2 = pauli_jordan_function(cs2)
        evals2 = np.linalg.eigvalsh(1j * iDelta2).real
        r_force_causal.append(spectral_gap_ratio(evals2))

        # Force element 0 || element N-1 (spacelike)
        to3 = TwoOrder(N, rng=np.random.default_rng(trial * 3000 + N * 11 + 2))
        if to3.u[0] > to3.u[N - 1]:
            to3.u[0], to3.u[N - 1] = to3.u[N - 1], to3.u[0]
        if to3.v[0] < to3.v[N - 1]:
            to3.v[0], to3.v[N - 1] = to3.v[N - 1], to3.v[0]
        cs3 = to3.to_causet()
        iDelta3 = pauli_jordan_function(cs3)
        evals3 = np.linalg.eigvalsh(1j * iDelta3).real
        r_force_spacelike.append(spectral_gap_ratio(evals3))

    mean_unc = np.mean(r_unconstrained)
    mean_caus = np.mean(r_force_causal)
    mean_space = np.mean(r_force_spacelike)

    print(f"  {N:>4} {mean_unc:>18.4f} {mean_caus:>18.4f} "
          f"{mean_space:>20.4f} {mean_caus - mean_unc:>12.4f} {mean_space - mean_unc:>16.4f}")

print(f"\n  INTERPRETATION: If dr ~ 0, GUE is robust to single constraints.")
print(f"  If dr > 0 (toward 1), the constraint pushes toward Poisson (localization).")
print(f"  If dr < 0, the constraint pushes toward even stronger repulsion.")

dt = time.time() - t0
print(f"\n  [Idea 764 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# IDEA 765: Master formula first correction for d=3
# ============================================================
print("\n" + "=" * 80)
print("IDEA 765: First d=3 correction to P[k|m] = 2(m-k)/[m(m+1)]")
print("=" * 80)
print("""
BACKGROUND: For 2-orders (d=2), the interval size distribution given a
"gap" m (separation in the first permutation) is:
   P[int=k | gap=m] = 2(m-k) / [m(m+1)]   for 0 <= k <= m-1

For d=3, we have THREE permutations. The gap m still refers to the first
permutation, but now i<j requires agreement in ALL THREE orderings.

QUESTION: What is P_3[k|m]? Can we write it as:
   P_3[k|m] = P_2[k|m] + eps * dP[k|m]
where eps captures the effect of the third ordering?

METHOD: Exact enumeration for small N, sampling for larger N.
""")
sys.stdout.flush()

t0 = time.time()

for d in [2, 3, 4]:
    print(f"\n  d = {d}:")
    N = 30
    n_trials = 500 if d <= 3 else 200

    gap_int_counts = {}
    gap_totals = {}

    for trial in range(n_trials):
        r = np.random.default_rng(trial * 5000 + d * 100)
        do = DOrder(d, N, rng=r)
        cs = do.to_causet_fast()

        perm0 = do.perms[0]
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if cs.order[i, j]:
                    gap = abs(perm0[j] - perm0[i])
                    interior = 0
                    for k in range(N):
                        if k == i or k == j:
                            continue
                        if cs.order[i, k] and cs.order[k, j]:
                            interior += 1

                    key = (gap, interior)
                    gap_int_counts[key] = gap_int_counts.get(key, 0) + 1
                    gap_totals[gap] = gap_totals.get(gap, 0) + 1

    print(f"    {'gap m':>6} {'k':>4} {'P(k|m) measured':>16} {'2-order formula':>16} {'correction':>12}")
    print("    " + "-" * 60)

    for m in [2, 3, 4, 5, 8]:
        if m not in gap_totals or gap_totals[m] == 0:
            continue
        total = gap_totals[m]
        for k in range(min(m, 6)):
            count = gap_int_counts.get((m, k), 0)
            p_measured = count / total
            p_formula = 2 * (m - k) / (m * (m + 1)) if k < m else 0
            correction = p_measured - p_formula
            if p_measured > 0.001 or k < 3:
                print(f"    {m:>6} {k:>4} {p_measured:>16.6f} {p_formula:>16.6f} {correction:>12.6f}")

print(f"\n  ANALYSIS: For d=3, the correction dP = P_3 - P_2 should reflect")
print(f"  the additional constraint from the third permutation.")
print(f"  If P_d[k|m] scales as ~(d-1)(m-k)/m^(d-1), the correction is multiplicative.")

dt = time.time() - t0
print(f"\n  [Idea 765 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# IDEA 766: Total order -> spacelike pairs. Plot <r> vs #spacelike.
# ============================================================
print("\n" + "=" * 80)
print("IDEA 766: Chain to antichain interpolation — <r> vs #spacelike pairs")
print("=" * 80)
print("""
BACKGROUND: A total order (chain) has all pairs causal, and its
Pauli-Jordan matrix has highly regular eigenvalues (<r> -> high).
A random 2-order has <r> ~ 0.60 (GUE).

INTERPOLATION: Start from a total order on N elements. Randomly "break"
relations to create spacelike pairs, one at a time. Track <r> as a
function of the fraction of pairs made spacelike.

This interpolates between chain (0% spacelike) and some disordered poset.
Where does the GUE transition happen?
""")
sys.stdout.flush()

t0 = time.time()

N = 30
print(f"  N = {N}")
print(f"  {'#spacelike':>10} {'frac spacelike':>15} {'f':>8} {'<r>':>8} {'std(r)':>8}")
print("  " + "-" * 55)

n_trials = 50
max_pairs = N * (N - 1) // 2

fractions = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

for frac in fractions:
    n_break = int(frac * max_pairs)
    r_values = []
    f_values = []

    for trial in range(n_trials):
        rl = np.random.default_rng(trial * 7000 + int(frac * 1000))

        order = np.zeros((N, N), dtype=bool)
        for i in range(N):
            for j in range(i + 1, N):
                order[i, j] = True

        causal_pairs = [(i, j) for i in range(N) for j in range(i + 1, N) if order[i, j]]

        if n_break > 0 and n_break <= len(causal_pairs):
            indices = rl.choice(len(causal_pairs), size=n_break, replace=False)
            for idx in indices:
                a, b = causal_pairs[idx]
                order[a, b] = False

        cs = FastCausalSet(N)
        cs.order = order

        f_val = ordering_fraction(cs)
        f_values.append(f_val)

        iDelta = pauli_jordan_function(cs)
        evals = np.linalg.eigvalsh(1j * iDelta).real
        r_val = spectral_gap_ratio(evals)
        r_values.append(r_val)

    actual_spacelike = int(n_break)
    print(f"  {actual_spacelike:>10} {frac:>15.3f} {np.mean(f_values):>8.4f} "
          f"{np.mean(r_values):>8.4f} {np.std(r_values):>8.4f}")

print(f"\n  NOTE: This is NOT the same as a 2-order interpolation; we're removing")
print(f"  random relations from a total order, which may not produce a valid poset")
print(f"  (transitivity can be violated). The <r> still measures spectral statistics.")

dt = time.time() - t0
print(f"\n  [Idea 766 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# IDEA 767: S(N/2) = c/3*ln(N). Remove one center element.
# ============================================================
print("\n" + "=" * 80)
print("IDEA 767: SJ entropy perturbation — remove one center element")
print("=" * 80)
print("""
BACKGROUND: The SJ entanglement entropy of the first half of a causal
set (by natural labeling) scales as S(N/2) ~ c/3 * ln(N) for 2-orders
(area law in 1+1D, where "area" = constant, so S ~ ln).

PERTURBATION: Remove the element closest to the center (the boundary
between the two halves). How does S change? In continuum, removing a
point from the boundary should decrease S (fewer entangled modes).

QUESTION: Is dS = S(with removal) - S(original) proportional to 1/N?
Is there a "bulk-boundary" correspondence where boundary elements
contribute more to entropy?
""")
sys.stdout.flush()

t0 = time.time()

print(f"  {'N':>4} {'S(N/2) original':>16} {'S after removal':>16} {'dS':>10} {'dS*N':>10}")
print("  " + "-" * 65)

for N in [12, 16, 20, 24, 30]:
    n_trials = 100 if N <= 20 else 50
    S_orig_list = []
    S_removed_list = []

    for trial in range(n_trials):
        r = np.random.default_rng(trial * 8000 + N * 13)
        cs, to = random_2order(N, rng_local=r)

        half = N // 2
        region_A = list(range(half))
        S_orig = sj_entropy(cs, region_A)
        S_orig_list.append(S_orig)

        elem_to_remove = half - 1

        keep = [i for i in range(N) if i != elem_to_remove]
        new_order = cs.order[np.ix_(keep, keep)]

        cs_reduced = FastCausalSet(N - 1)
        cs_reduced.order = new_order

        new_region_A = list(range(half - 1))
        if len(new_region_A) > 0:
            S_removed = sj_entropy(cs_reduced, new_region_A)
        else:
            S_removed = 0.0
        S_removed_list.append(S_removed)

    mean_orig = np.mean(S_orig_list)
    mean_removed = np.mean(S_removed_list)
    delta_S = mean_removed - mean_orig

    print(f"  {N:>4} {mean_orig:>16.6f} {mean_removed:>16.6f} "
          f"{delta_S:>10.6f} {delta_S * N:>10.4f}")

print(f"\n  INTERPRETATION: If dS*N -> const as N->inf, each boundary element")
print(f"  contributes O(1/N) to the entropy — consistent with area law.")
print(f"  If dS = const (independent of N), boundary elements carry O(1) entropy.")

dt = time.time() - t0
print(f"\n  [Idea 767 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# IDEA 768: Fiedler value under highest-degree removal
# ============================================================
print("\n" + "=" * 80)
print("IDEA 768: Fiedler value perturbation — remove highest-degree element")
print("=" * 80)
print("""
BACKGROUND: The Fiedler value lambda_2 (second-smallest eigenvalue of the
graph Laplacian of the Hasse diagram) measures algebraic connectivity.
For random 2-orders, lambda_2 scales approximately as O(1) for moderate N.

PERTURBATION: Remove the element with the highest degree in the Hasse
diagram. This is the most "connected" element. How does lambda_2 change?
Compare with removing a random element.

QUESTION: Is the Hasse diagram robust or fragile to targeted removal?
""")
sys.stdout.flush()

t0 = time.time()

print(f"  {'N':>4} {'lam2 original':>14} {'lam2 (rm max-deg)':>18} {'lam2 (rm random)':>18} "
      f"{'d_lam2 (max-deg)':>18} {'d_lam2 (random)':>16}")
print("  " + "-" * 90)

for N in [20, 30, 50, 80, 100]:
    n_trials = 100 if N <= 50 else 50

    lam2_orig_list = []
    lam2_max_deg_list = []
    lam2_random_list = []

    for trial in range(n_trials):
        r = np.random.default_rng(trial * 9000 + N * 17)
        cs, to = random_2order(N, rng_local=r)

        L = hasse_laplacian(cs)
        evals = eigvalsh(L)
        lam2_orig = evals[1]
        lam2_orig_list.append(lam2_orig)

        adj = hasse_adjacency(cs)
        degrees = np.sum(adj, axis=1).astype(int)
        max_deg_elem = np.argmax(degrees)

        keep = [i for i in range(N) if i != max_deg_elem]
        links_reduced = cs.link_matrix()[np.ix_(keep, keep)]
        adj_reduced = (links_reduced | links_reduced.T).astype(np.float64)
        deg_reduced = np.sum(adj_reduced, axis=1)
        L_reduced = np.diag(deg_reduced) - adj_reduced
        evals_reduced = eigvalsh(L_reduced)
        lam2_max_deg = evals_reduced[1]
        lam2_max_deg_list.append(lam2_max_deg)

        rand_elem = r.integers(N)
        keep2 = [i for i in range(N) if i != rand_elem]
        links_reduced2 = cs.link_matrix()[np.ix_(keep2, keep2)]
        adj_reduced2 = (links_reduced2 | links_reduced2.T).astype(np.float64)
        deg_reduced2 = np.sum(adj_reduced2, axis=1)
        L_reduced2 = np.diag(deg_reduced2) - adj_reduced2
        evals_reduced2 = eigvalsh(L_reduced2)
        lam2_random = evals_reduced2[1]
        lam2_random_list.append(lam2_random)

    m_orig = np.mean(lam2_orig_list)
    m_max = np.mean(lam2_max_deg_list)
    m_rand = np.mean(lam2_random_list)

    print(f"  {N:>4} {m_orig:>14.4f} {m_max:>18.4f} {m_rand:>18.4f} "
          f"{m_max - m_orig:>18.4f} {m_rand - m_orig:>16.4f}")

print(f"\n  INTERPRETATION: If d_lam2 < 0 (max-deg removal DECREASES connectivity),")
print(f"  the network is FRAGILE to targeted attack. If d_lam2 > 0 for random")
print(f"  removal but < 0 for targeted, this mimics scale-free network behavior.")

dt = time.time() - t0
print(f"\n  [Idea 768 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# IDEA 769: Exact Z(beta) at N=4, perturbation to N=5
# ============================================================
print("\n" + "=" * 80)
print("IDEA 769: Partition function Z(beta) — exact at N=4, perturbation to N=5")
print("=" * 80)
print("""
BACKGROUND: The partition function Z(beta) = sum_C exp(-beta*S[C]) sums over
all posets (or 2-orders) on N elements, weighted by the BD action.

For N=4 elements, we can enumerate ALL 2-orders (4! x 4! = 576 pairs
of permutations, many giving the same poset). Z(beta) is exactly computable.

PERTURBATION: At N=5, there are 5!x5! = 14400 pairs. Still enumerable.
Write Z_5(beta) = Z_4(beta) * (1 + d1*beta + d2*beta^2 + ...).
Does the perturbation series in 1/N converge?
""")
sys.stdout.flush()

t0 = time.time()

eps_bd = 0.5

from itertools import permutations as iter_perms

for N_enum in [4, 5]:
    print(f"\n  N = {N_enum}: Enumerating all {np.math.factorial(N_enum)**2} permutation pairs...")
    sys.stdout.flush()

    actions = []
    all_perms = list(iter_perms(range(N_enum)))

    for u in all_perms:
        u_arr = np.array(u)
        for v in all_perms:
            v_arr = np.array(v)
            to = TwoOrder.from_permutations(u_arr, v_arr)
            cs = to.to_causet()
            S = bd_action_corrected(cs, eps_bd)
            actions.append(S)

    actions = np.array(actions)
    print(f"    Total configs: {len(actions)}")
    print(f"    Unique actions: {len(np.unique(np.round(actions, 6)))}")
    print(f"    Action range: [{np.min(actions):.4f}, {np.max(actions):.4f}]")
    print(f"    Mean action: {np.mean(actions):.6f}")

    betas = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    print(f"    {'beta':>8} {'Z(beta)':>16} {'<S>':>12} {'<S^2>-<S>^2':>14} {'F=-ln(Z)/beta':>14}")
    print("    " + "-" * 70)

    Z_values = {}
    for beta in betas:
        weights = np.exp(-beta * actions)
        Z = np.sum(weights)
        mean_S = np.sum(actions * weights) / Z
        mean_S2 = np.sum(actions**2 * weights) / Z
        var_S = mean_S2 - mean_S**2
        F = -np.log(Z) / beta if beta > 0 else 0.0
        Z_values[beta] = Z
        print(f"    {beta:>8.1f} {Z:>16.4f} {mean_S:>12.6f} {var_S:>14.6f} {F:>14.6f}")

    if N_enum == 4:
        Z4 = Z_values.copy()
    else:
        Z5 = Z_values.copy()

print(f"\n  PERTURBATION ANALYSIS: Z_5(beta) / Z_4(beta)")
print(f"  {'beta':>8} {'Z_4':>14} {'Z_5':>14} {'Z_5/Z_4':>12} {'ln(Z5/Z4)':>12}")
print("  " + "-" * 65)
for beta in betas:
    ratio = Z5[beta] / Z4[beta]
    log_ratio = np.log(ratio)
    print(f"  {beta:>8.1f} {Z4[beta]:>14.4f} {Z5[beta]:>14.4f} "
          f"{ratio:>12.4f} {log_ratio:>12.6f}")

print(f"\n  If Z_(N+1)/Z_N is smooth in beta, the partition function is")
print(f"  perturbatively well-behaved as N grows.")

dt = time.time() - t0
print(f"\n  [Idea 769 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# IDEA 770: Antichain ~ 2*sqrt(N). Force one causal pair spacelike.
# ============================================================
print("\n" + "=" * 80)
print("IDEA 770: Antichain perturbation — force one causal pair spacelike")
print("=" * 80)
print("""
BACKGROUND: For a random 2-order on N elements, the longest antichain
(maximum set of mutually spacelike elements) scales as ~ 2*sqrt(N).

PERTURBATION: Take a random 2-order. Find a CAUSAL pair (i,j) with
i < j. Force them to be SPACELIKE (remove the relation). This breaks
the 2-order structure but creates a valid poset. Does the antichain
grow? By how much?

ALSO: Force a LINK (irreducible relation) to become spacelike.
This should have a bigger effect since links are "load-bearing".
""")
sys.stdout.flush()

t0 = time.time()

print(f"  {'N':>4} {'antichain orig':>14} {'anti (rm causal)':>18} {'anti (rm link)':>16} "
      f"{'d (causal)':>12} {'d (link)':>10} {'2*sqrt(N)':>10}")
print("  " + "-" * 90)

for N in [16, 25, 36, 49, 64]:
    n_trials = 100 if N <= 36 else 50

    anti_orig_list = []
    anti_rm_causal_list = []
    anti_rm_link_list = []

    for trial in range(n_trials):
        r = np.random.default_rng(trial * 10000 + N * 19)
        cs, to = random_2order(N, rng_local=r)

        a_orig = longest_antichain(cs)
        anti_orig_list.append(a_orig)

        # Find a random causal pair
        causal_pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                if cs.order[i, j]:
                    causal_pairs.append((i, j))

        if len(causal_pairs) > 0:
            idx = r.integers(len(causal_pairs))
            a, b = causal_pairs[idx]

            new_order = cs.order.copy()
            new_order[a, b] = False
            cs_mod = FastCausalSet(N)
            cs_mod.order = new_order
            a_rm = longest_antichain(cs_mod)
            anti_rm_causal_list.append(a_rm)

        # Find a random link
        links = cs.link_matrix()
        link_pairs = list(zip(*np.where(links)))

        if len(link_pairs) > 0:
            idx = r.integers(len(link_pairs))
            a, b = link_pairs[idx]

            new_order2 = cs.order.copy()
            new_order2[a, b] = False
            cs_mod2 = FastCausalSet(N)
            cs_mod2.order = new_order2
            a_rm_link = longest_antichain(cs_mod2)
            anti_rm_link_list.append(a_rm_link)

    m_orig = np.mean(anti_orig_list)
    m_causal = np.mean(anti_rm_causal_list) if anti_rm_causal_list else 0
    m_link = np.mean(anti_rm_link_list) if anti_rm_link_list else 0

    print(f"  {N:>4} {m_orig:>14.2f} {m_causal:>18.2f} {m_link:>16.2f} "
          f"{m_causal - m_orig:>12.4f} {m_link - m_orig:>10.4f} {2 * np.sqrt(N):>10.2f}")

print(f"\n  INTERPRETATION: If d(link) >> d(causal), links are 'load-bearing'")
print(f"  for the causal structure — removing a link creates more spacelike room.")
print(f"  If d ~ 0, the antichain is robust to single-relation perturbations.")

dt = time.time() - t0
print(f"\n  [Idea 770 completed in {dt:.1f}s]")
sys.stdout.flush()


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY: PERTURBATION THEORY RESULTS")
print("=" * 80)
print("""
761. E[f] perturbation:     Adding one relation -> delta_f from cascade through transitivity
762. E[links] removal:      Element removal preserves/breaks 2-order universality?
763. CDT Kronecker:         One intra-slice relation -> n_pos shift (stability of SJ vacuum)
764. GUE <r>:               One constraint -> shift toward Poisson or stable GUE?
765. Master formula d=3:    Correction to P[k|m] from third permutation
766. Chain->disorder:       <r> interpolation from total order to disordered poset
767. SJ entropy removal:    dS from boundary element removal — bulk-boundary?
768. Fiedler targeted:      Hub removal -> fragile or robust algebraic connectivity?
769. Z(beta) perturbation:  N=4->N=5 partition function — convergent series?
770. Antichain stability:   Link removal vs general relation removal
""")
print("=" * 80)
print("Experiment 125 complete.")

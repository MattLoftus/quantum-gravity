"""
Experiment 103: Paper G Exact Results Expansion — Ideas 541-550

Derive MORE EXACT RESULTS for the exact combinatorics paper.
Focus: exact enumeration for N=4-8 and analytic derivation.

Ideas:
541. E[number of MAXIMAL ELEMENTS] for a random 2-order
542. E[number of MINIMAL ELEMENTS] (by symmetry = maximal)
543. Exact CORRELATION between chain length and ordering fraction
544. E[number of CONNECTED COMPONENTS of the Hasse diagram] / exact P(connected)
545. Exact CHROMATIC NUMBER of the Hasse diagram for small N
546. AUTOMORPHISM GROUP SIZE of a random 2-order for N=4,5,6
547. E[number of 2-ELEMENT ANTICHAINS] (spacelike pairs)
548. E[number of k-ELEMENT ANTICHAINS] for general k
549. JOINT DISTRIBUTION of (ordering fraction, link fraction) for N=4,5
550. TUTTE POLYNOMIAL of the Hasse diagram for small N
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from itertools import permutations, combinations
from collections import Counter, defaultdict
from math import factorial, comb, gcd
from fractions import Fraction

# =============================================================================
# Core: enumerate all 2-orders for small N
# =============================================================================

def build_2order(u, v, N):
    """Build the partial order from two permutations u, v.
    i < j iff u[i] < u[j] and v[i] < v[j].
    Returns adjacency matrix: order[i][j] = 1 if i < j.
    """
    order = [[False]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j and u[i] < u[j] and v[i] < v[j]:
                order[i][j] = True
    return order

def transitive_reduction(order, N):
    """Compute the Hasse diagram (transitive reduction).
    link[i][j] = True iff i < j and there's no k with i < k < j.
    """
    link = [[False]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if order[i][j]:
                # Check if there's an intermediate element
                has_intermediate = False
                for k in range(N):
                    if k != i and k != j and order[i][k] and order[k][j]:
                        has_intermediate = True
                        break
                if not has_intermediate:
                    link[i][j] = True
    return link

def get_hasse_undirected_edges(link, N):
    """Get undirected edges of the Hasse diagram."""
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if link[i][j] or link[j][i]:
                edges.append((i, j))
    return edges

def is_connected_undirected(edges, N):
    """Check if the undirected graph on N vertices is connected."""
    if N <= 1:
        return True
    adj = defaultdict(set)
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    visited = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for nb in adj[node]:
            if nb not in visited:
                stack.append(nb)
    return len(visited) == N

def count_connected_components(edges, N):
    """Count connected components of undirected graph."""
    adj = defaultdict(set)
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    visited = set()
    components = 0
    for start in range(N):
        if start not in visited:
            components += 1
            stack = [start]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                for nb in adj[node]:
                    if nb not in visited:
                        stack.append(nb)
    return components

def longest_chain(order, N):
    """Find the length of the longest chain (number of elements)."""
    # DP on topological order
    # chain_len[i] = longest chain ending at i
    chain_len = [1]*N
    # Sort by u-rank for topological order (any consistent ordering works)
    for j in range(N):
        for i in range(N):
            if order[i][j]:
                chain_len[j] = max(chain_len[j], chain_len[i] + 1)
    return max(chain_len)

def chromatic_number_greedy_exact(edges, N):
    """Compute exact chromatic number by trying all colorings.
    For small N this is feasible.
    """
    if not edges:
        return 1 if N > 0 else 0

    adj = defaultdict(set)
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)

    # Try k colors starting from 1
    for k in range(1, N+1):
        if can_color(adj, N, k):
            return k
    return N

def can_color(adj, N, k):
    """Check if graph can be colored with k colors via backtracking."""
    colors = [-1]*N

    def backtrack(node):
        if node == N:
            return True
        for c in range(k):
            if all(colors[nb] != c for nb in adj[node] if colors[nb] != -1):
                colors[node] = c
                if backtrack(node + 1):
                    return True
                colors[node] = -1
        return False

    return backtrack(0)

def compute_automorphisms(order, N):
    """Count automorphisms of the poset.
    An automorphism is a permutation p of {0,...,N-1} such that
    order[i][j] = order[p[i]][p[j]] for all i,j.
    """
    count = 0
    for perm in permutations(range(N)):
        is_auto = True
        for i in range(N):
            for j in range(N):
                if order[i][j] != order[perm[i]][perm[j]]:
                    is_auto = False
                    break
            if not is_auto:
                break
        if is_auto:
            count += 1
    return count

def tutte_poly_cached(edges_frozen, memo={}):
    """Compute Tutte polynomial T(x,y) via deletion-contraction with memoization.
    edges_frozen: frozenset of (min,max) edge tuples.
    Returns dict {(i,j): coeff} for T(x,y) = sum coeff * x^i * y^j.
    """
    if edges_frozen in memo:
        return memo[edges_frozen]

    edges = list(edges_frozen)

    if not edges:
        memo[edges_frozen] = {(0, 0): 1}
        return memo[edges_frozen]

    # Get vertices from edges only (no isolated vertices tracked)
    vertices = set()
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        vertices.add(u)
        vertices.add(v)

    # Find connected components
    visited = set()
    components = []
    for start in sorted(vertices):
        if start in visited:
            continue
        comp_v = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            comp_v.add(node)
            for nb in adj[node]:
                if nb not in visited:
                    stack.append(nb)
        comp_e = frozenset((u, v) for u, v in edges if u in comp_v)
        components.append(comp_e)

    if len(components) > 1:
        result = {(0, 0): 1}
        for comp_e in components:
            comp_T = tutte_poly_cached(comp_e, memo)
            new_result = {}
            for (i1, j1), c1 in result.items():
                for (i2, j2), c2 in comp_T.items():
                    key = (i1 + i2, j1 + j2)
                    new_result[key] = new_result.get(key, 0) + c1 * c2
            result = new_result
        memo[edges_frozen] = result
        return result

    # Single connected component -- pick an edge
    e = edges[0]
    rest_list = edges[1:]
    rest_frozen = frozenset(rest_list)
    u_e, v_e = e

    # Check if bridge: does removing e disconnect?
    is_bridge = False
    if rest_list:
        rest_adj = defaultdict(set)
        for a, b in rest_list:
            rest_adj[a].add(b)
            rest_adj[b].add(a)
        visited2 = set()
        stack = [u_e]
        while stack:
            node = stack.pop()
            if node in visited2:
                continue
            visited2.add(node)
            for nb in rest_adj[node]:
                if nb not in visited2:
                    stack.append(nb)
        is_bridge = (v_e not in visited2)
    else:
        is_bridge = True  # single edge is a bridge

    # Delete: just remove the edge
    T_del = tutte_poly_cached(rest_frozen, memo)

    # Contract: merge v_e into u_e
    new_edges = set()
    for (a, b) in rest_list:
        na = u_e if a == v_e else a
        nb = u_e if b == v_e else b
        if na != nb:
            new_edges.add((min(na, nb), max(na, nb)))
    T_con = tutte_poly_cached(frozenset(new_edges), memo)

    if is_bridge:
        result = {}
        for (i, j), c in T_con.items():
            key = (i + 1, j)
            result[key] = result.get(key, 0) + c
    else:
        result = {}
        for (i, j), c in T_del.items():
            result[(i, j)] = result.get((i, j), 0) + c
        for (i, j), c in T_con.items():
            result[(i, j)] = result.get((i, j), 0) + c

    memo[edges_frozen] = result
    return result


# =============================================================================
# Main enumeration
# =============================================================================

def enumerate_all_2orders(N):
    """Enumerate all (N!)^2 2-orders and compute properties."""
    perms = list(permutations(range(N)))
    n_perms = len(perms)
    total = n_perms * n_perms

    print(f"\n{'='*70}")
    print(f"N = {N}: Enumerating all {total} 2-orders ({n_perms} x {n_perms} permutation pairs)")
    print(f"{'='*70}")

    # Accumulators
    maximal_counts = []
    minimal_counts = []
    chain_lengths = []
    ordering_fractions = []
    link_fractions = []
    connected_count = 0
    component_counts = []
    chromatic_numbers = []
    automorphism_sizes = []
    antichain_2_counts = []
    antichain_k_counts = {k: [] for k in range(1, N+1)}

    # For joint distribution
    of_lf_pairs = []

    # For Tutte polynomial: accumulate average
    tutte_accum = defaultdict(int)
    tutte_count = 0

    # For correlation
    chain_of_pairs = []  # (chain_length, ordering_fraction) pairs

    for ui, u in enumerate(perms):
        if ui % max(1, n_perms // 10) == 0:
            print(f"  Progress: u-perm {ui}/{n_perms}")
        for v in perms:
            order = build_2order(u, v, N)

            # --- Maximal elements (no element above) ---
            maximal = 0
            minimal = 0
            for i in range(N):
                is_maximal = True
                is_minimal = True
                for j in range(N):
                    if order[i][j]:  # i < j means i is not maximal
                        is_maximal = False
                    if order[j][i]:  # j < i means i is not minimal
                        is_minimal = False
                maximal += is_maximal
                minimal += is_minimal
            maximal_counts.append(maximal)
            minimal_counts.append(minimal)

            # --- Ordering fraction ---
            R = sum(1 for i in range(N) for j in range(N) if i != j and order[i][j])
            f_ord = R / (N * (N - 1))  # ordering fraction (fraction of ordered DIRECTED pairs)
            ordering_fractions.append(f_ord)

            # --- Chain length ---
            cl = longest_chain(order, N)
            chain_lengths.append(cl)
            chain_of_pairs.append((cl, f_ord))

            # --- Hasse diagram ---
            link = transitive_reduction(order, N)
            edges = get_hasse_undirected_edges(link, N)

            # --- Links count ---
            n_links = len(edges)
            link_frac = n_links / max(1, R) if R > 0 else 1.0
            link_fractions.append(link_frac)
            of_lf_pairs.append((f_ord, n_links))

            # --- Connected components ---
            n_comp = count_connected_components(edges, N)
            component_counts.append(n_comp)
            if n_comp == 1:
                connected_count += 1

            # --- Chromatic number (only for small N) ---
            if N <= 6:
                chi = chromatic_number_greedy_exact(edges, N)
                chromatic_numbers.append(chi)

            # --- Automorphism group (only for small N) ---
            if N <= 5:
                aut = compute_automorphisms(order, N)
                automorphism_sizes.append(aut)

            # --- k-element antichains ---
            # Two elements form an antichain iff they are spacelike
            # (neither i<j nor j<i)
            spacelike_pairs = 0
            for i in range(N):
                for j in range(i+1, N):
                    if not order[i][j] and not order[j][i]:
                        spacelike_pairs += 1
            antichain_2_counts.append(spacelike_pairs)

            # General k-antichains
            for k in range(1, N+1):
                count_k = 0
                if k == 1:
                    count_k = N
                elif k == 2:
                    count_k = spacelike_pairs
                else:
                    for subset in combinations(range(N), k):
                        is_antichain = True
                        for a, b in combinations(subset, 2):
                            if order[a][b] or order[b][a]:
                                is_antichain = False
                                break
                        if is_antichain:
                            count_k += 1
                antichain_k_counts[k].append(count_k)

            # --- Tutte polynomial (only for N <= 5) ---
            if N <= 5:
                if edges:
                    edge_set = frozenset((min(a,b), max(a,b)) for a,b in edges)
                    T = tutte_poly_cached(edge_set)
                    for key, val in T.items():
                        tutte_accum[key] += val
                else:
                    tutte_accum[(0, 0)] = tutte_accum.get((0, 0), 0) + 1
                tutte_count += 1

    # --- Report results ---
    print(f"\n--- RESULTS FOR N = {N} ---")
    print(f"Total 2-orders enumerated: {total}")

    # Idea 541: E[maximal elements]
    E_max = Fraction(sum(maximal_counts), total)
    print(f"\n[Idea 541] E[maximal elements] = {float(E_max):.6f} = {E_max}")
    max_dist = Counter(maximal_counts)
    print(f"  Distribution: {dict(sorted(max_dist.items()))}")

    # Idea 542: E[minimal elements]
    E_min = Fraction(sum(minimal_counts), total)
    print(f"\n[Idea 542] E[minimal elements] = {float(E_min):.6f} = {E_min}")
    min_dist = Counter(minimal_counts)
    print(f"  Distribution: {dict(sorted(min_dist.items()))}")
    print(f"  Symmetry check: E[max] == E[min]? {E_max == E_min}")

    # Idea 543: Correlation(chain length, ordering fraction)
    cl_arr = np.array(chain_lengths, dtype=float)
    of_arr = np.array(ordering_fractions, dtype=float)
    corr = np.corrcoef(cl_arr, of_arr)[0, 1]
    print(f"\n[Idea 543] Correlation(chain length, ordering fraction) = {corr:.6f}")
    print(f"  E[chain length] = {np.mean(cl_arr):.4f}")
    print(f"  Std[chain length] = {np.std(cl_arr):.4f}")
    print(f"  Chain length distribution: {dict(sorted(Counter(chain_lengths).items()))}")
    # Covariance
    cov = np.cov(cl_arr, of_arr)[0, 1]
    print(f"  Cov(chain, ordering) = {cov:.6f}")

    # Idea 544: Connected components
    E_comp = Fraction(sum(component_counts), total)
    P_connected = Fraction(connected_count, total)
    print(f"\n[Idea 544] E[connected components of Hasse] = {float(E_comp):.6f} = {E_comp}")
    print(f"  P(Hasse connected) = {float(P_connected):.6f} = {P_connected}")
    comp_dist = Counter(component_counts)
    print(f"  Component distribution: {dict(sorted(comp_dist.items()))}")

    # Idea 545: Chromatic number
    if chromatic_numbers:
        E_chi = np.mean(chromatic_numbers)
        chi_dist = Counter(chromatic_numbers)
        print(f"\n[Idea 545] E[chromatic number of Hasse] = {E_chi:.6f}")
        print(f"  Distribution: {dict(sorted(chi_dist.items()))}")

    # Idea 546: Automorphism group
    if automorphism_sizes:
        E_aut = Fraction(sum(automorphism_sizes), total)
        print(f"\n[Idea 546] E[|Aut|] = {float(E_aut):.6f} = {E_aut}")
        aut_dist = Counter(automorphism_sizes)
        print(f"  Distribution: {dict(sorted(aut_dist.items()))}")

    # Idea 547: 2-element antichains
    E_ac2 = Fraction(sum(antichain_2_counts), total)
    print(f"\n[Idea 547] E[2-element antichains] = {float(E_ac2):.6f} = {E_ac2}")
    predicted = Fraction(N * (N - 1), 4)
    print(f"  Predicted N(N-1)/4 = {float(predicted):.6f} = {predicted}")
    print(f"  Match? {E_ac2 == predicted}")

    # Idea 548: k-element antichains
    print(f"\n[Idea 548] E[k-element antichains]:")
    for k in range(1, N+1):
        E_ack = Fraction(sum(antichain_k_counts[k]), total)
        print(f"  k={k}: E = {float(E_ack):.6f} = {E_ack}")

    # Idea 549: Joint distribution (ordering fraction, link count)
    if N <= 5:
        print(f"\n[Idea 549] Joint distribution of (ordering fraction, link count):")
        joint = Counter(of_lf_pairs)
        # Group by ordering fraction
        of_values = sorted(set(x[0] for x in of_lf_pairs))
        for of_val in of_values:
            links_at_of = [(lc, cnt) for (of_v, lc), cnt in joint.items() if of_v == of_val]
            links_at_of.sort()
            total_at_of = sum(c for _, c in links_at_of)
            link_vals = [lc for lc, _ in links_at_of]
            E_links_given_of = sum(lc * cnt for lc, cnt in links_at_of) / total_at_of
            print(f"  f={of_val:.4f} (count={total_at_of}): E[links|f]={E_links_given_of:.3f}, "
                  f"link values={link_vals}")

    # Idea 550: Tutte polynomial
    if N <= 5 and tutte_accum:
        print(f"\n[Idea 550] Average Tutte polynomial coefficients (sum over all 2-orders):")
        sorted_keys = sorted(tutte_accum.keys())
        for key in sorted_keys:
            val = Fraction(tutte_accum[key], total)
            if abs(float(val)) > 1e-10:
                print(f"  x^{key[0]} * y^{key[1]}: {float(val):.6f} (= {val})")

    return {
        'E_max': E_max, 'E_min': E_min,
        'corr_chain_of': corr,
        'E_components': E_comp, 'P_connected': P_connected,
        'E_ac2': E_ac2,
        'antichain_k': {k: Fraction(sum(antichain_k_counts[k]), total) for k in range(1, N+1)},
    }


# =============================================================================
# Analytic derivations
# =============================================================================

def analytic_results():
    """Derive analytic formulas and verify against enumeration."""
    print("\n" + "="*70)
    print("ANALYTIC DERIVATIONS")
    print("="*70)

    # --- Idea 541/542: E[maximal elements] ---
    print("\n--- Idea 541/542: E[maximal/minimal elements] ---")
    print("""
THEOREM (Expected number of maximal elements):
For a random 2-order on N elements, E[maximal] = E[minimal] = N * P(element i is maximal).

Element i is maximal iff no j has both u_j > u_i and v_j > v_i.
Equivalently, i is maximal iff for ALL other elements j, either u_j < u_i or v_j < v_i.

Fix element i. WLOG by symmetry, consider P(element 0 is maximal).
If element 0 has u-rank r (so r elements have smaller u-value, N-1-r have larger),
then element 0 is maximal iff ALL elements with u-rank > r have v-rank < v_0.

Given u-rank r for element 0 (0-indexed), there are N-1-r elements above it in u.
Element 0 has v-rank s. For element 0 to be maximal, all N-1-r elements with
higher u-rank must have v-rank < s.

P(maximal | u-rank=r, v-rank=s) = C(s, N-1-r) / C(N-1, N-1-r) if s >= N-1-r, else 0.

Actually, simpler approach:
P(element i is maximal) = Sum over positions. Let's compute directly.

In a random 2-order, element i has u-rank and v-rank uniformly and independently
distributed. Element i is maximal iff it is NOT dominated by any other element.

By inclusion-exclusion or direct calculation:
P(maximal) = Sum_{r=0}^{N-1} (1/N) * [r/(N-1)]^1 ...

Actually, the cleanest: element i is a maximum of the 2D point set
(u_i, v_i) in {1,...,N}^2 (with distinct values in each coordinate).
This is equivalent to: in a random permutation sigma = v o u^{-1},
element at position r has value sigma(r). It is maximal iff no element
to its right has a higher value.

After re-indexing by u, element at u-position r (0-indexed from 0 to N-1)
is maximal iff sigma(r) > sigma(j) for all j > r. This means sigma(r) is
the maximum of {sigma(r), sigma(r+1), ..., sigma(N-1)}.

P(sigma(r) = max of last N-r values) = 1/(N-r).

So P(element with u-rank r is maximal) = 1/(N-r).

E[maximal] = Sum_{r=0}^{N-1} 1/(N-r) = Sum_{k=1}^{N} 1/k = H_N.

RESULT: E[number of maximal elements] = H_N (the N-th harmonic number).
By symmetry, E[number of minimal elements] = H_N.
""")

    for N in range(2, 9):
        H_N = sum(Fraction(1, k) for k in range(1, N+1))
        print(f"  N={N}: H_N = {H_N} = {float(H_N):.6f}")

    # --- Idea 547: 2-element antichains ---
    print("\n--- Idea 547: E[2-element antichains] = N(N-1)/4 ---")
    print("""
THEOREM: E[number of 2-element antichains] = N(N-1)/4.

Proof: For any pair (i,j), P(spacelike) = 1 - P(i<j) - P(j<i) = 1 - 1/4 - 1/4 = 1/2.
Number of pairs = C(N,2) = N(N-1)/2.
E[spacelike pairs] = N(N-1)/2 * 1/2 = N(N-1)/4.
""")

    # --- Idea 548: k-element antichains ---
    print("\n--- Idea 548: E[k-element antichains] ---")
    print("""
THEOREM: E[number of k-element antichains] = C(N,k) * P(k elements are mutually spacelike).

For k elements to form an antichain, ALL C(k,2) pairs must be spacelike.
In a 2-order, elements {e_1,...,e_k} are mutually spacelike iff when sorted by u-rank,
their v-values form a DECREASING sequence.

P(k elements form antichain) = (number of permutations of k with longest decreasing
subsequence = k) / k! = 1/k!   ... No, that's too simple.

Actually: Fix k elements. Their u-ranks and v-ranks each form a uniformly random
sub-permutation of size k. They are all mutually spacelike iff the composed
sub-permutation is the REVERSE permutation (completely decreasing).

P(all k mutually spacelike) = 1/k! (probability that a random permutation of k
elements is the reverse permutation? No, that's 1/k!.)

Wait. The k elements have some sub-permutation sigma in their v-values when
sorted by u-values. For ALL pairs to be spacelike, sigma must be strictly
decreasing, i.e., sigma = (k, k-1, ..., 1). There is exactly 1 such permutation
out of k!. So P = 1/k!.

E[k-antichains] = C(N,k) / k! = N! / (k!)^2 / (N-k)!   ...

Wait: C(N,k) * (1/k!) = N! / (k! * (N-k)!) * (1/k!) = N! / ((k!)^2 * (N-k)!).

Hmm, let me reconsider. The k elements' u-sub-ranking is a fixed ordering.
Their v-sub-ranking is a uniform random permutation of {1,...,k}.
They are all spacelike iff in u-order, v is strictly decreasing.
Probability = 1/k!.

So E[k-antichains] = C(N,k) / k!.
""")

    for N in range(2, 9):
        print(f"  N={N}:")
        for k in range(1, N+1):
            E_ack = Fraction(factorial(N), factorial(k)**2 * factorial(N - k))
            print(f"    k={k}: C(N,k)/k! = {E_ack} = {float(E_ack):.6f}")

    # --- Idea 544: P(Hasse connected) ---
    print("\n--- Idea 544: P(Hasse diagram connected) ---")
    print("""
For the Hasse diagram to be disconnected, there must exist a non-trivial partition
of elements into sets A and B such that no link connects A to B. This is rarer than
the poset being disconnected (which requires no RELATION between A and B).

The Hasse diagram is disconnected iff the poset has no link between some components.
A poset can have relations across components that aren't links (they go through
intermediate elements), but the Hasse diagram only has links.

For N=2: There are 4 2-orders. In 2, elements are related (1 link); in 2, they're
unrelated (0 links, disconnected). P(connected) = 2/4 = 1/2.

For larger N, we compute exactly below.
""")


def verify_analytic_maximal(results_dict):
    """Verify analytic formula E[maximal] = H_N against exact enumeration."""
    print("\n--- Verification: E[maximal] = H_N ---")
    for N, res in sorted(results_dict.items()):
        H_N = sum(Fraction(1, k) for k in range(1, N+1))
        match = (res['E_max'] == H_N)
        print(f"  N={N}: exact={res['E_max']} = {float(res['E_max']):.6f}, "
              f"H_N={H_N} = {float(H_N):.6f}, match={match}")


def verify_antichain_formula(results_dict):
    """Verify E[k-antichains] = C(N,k)/k!"""
    print("\n--- Verification: E[k-antichains] = C(N,k)/k! ---")
    for N, res in sorted(results_dict.items()):
        print(f"  N={N}:")
        for k in range(1, N+1):
            exact = res['antichain_k'][k]
            predicted = Fraction(factorial(N), factorial(k)**2 * factorial(N - k))
            match = (exact == predicted)
            print(f"    k={k}: exact={exact}, C(N,k)/k!={predicted}, match={match}")


# =============================================================================
# Monte Carlo for larger N
# =============================================================================

def monte_carlo_larger_N(N, n_samples=100000):
    """Monte Carlo estimation for N=7,8 and larger."""
    print(f"\n{'='*70}")
    print(f"Monte Carlo for N={N} ({n_samples} samples)")
    print(f"{'='*70}")

    maximal_counts = []
    minimal_counts = []
    chain_lengths = []
    ordering_fracs = []
    component_counts = []
    antichain_2_counts = []
    connected_count = 0

    for _ in range(n_samples):
        u = np.random.permutation(N)
        v = np.random.permutation(N)

        # Build order
        order = [[False]*N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i != j and u[i] < u[j] and v[i] < v[j]:
                    order[i][j] = True

        # Maximal/minimal
        maximal = sum(1 for i in range(N) if not any(order[i][j] for j in range(N)))
        minimal = sum(1 for i in range(N) if not any(order[j][i] for j in range(N)))
        maximal_counts.append(maximal)
        minimal_counts.append(minimal)

        # Ordering fraction
        R = sum(1 for i in range(N) for j in range(N) if i != j and order[i][j])
        f_ord = R / (N * (N - 1))
        ordering_fracs.append(f_ord)

        # Chain length
        cl = longest_chain(order, N)
        chain_lengths.append(cl)

        # Hasse diagram
        link = transitive_reduction(order, N)
        edges = get_hasse_undirected_edges(link, N)
        n_comp = count_connected_components(edges, N)
        component_counts.append(n_comp)
        if n_comp == 1:
            connected_count += 1

        # 2-antichains
        sp = sum(1 for i in range(N) for j in range(i+1, N) if not order[i][j] and not order[j][i])
        antichain_2_counts.append(sp)

    H_N = sum(1.0/k for k in range(1, N+1))

    print(f"  E[maximal]: {np.mean(maximal_counts):.4f} (predicted H_N = {H_N:.4f})")
    print(f"  E[minimal]: {np.mean(minimal_counts):.4f} (predicted H_N = {H_N:.4f})")
    print(f"  E[2-antichains]: {np.mean(antichain_2_counts):.4f} (predicted N(N-1)/4 = {N*(N-1)/4:.4f})")
    print(f"  Corr(chain, ordering): {np.corrcoef(chain_lengths, ordering_fracs)[0,1]:.4f}")
    print(f"  E[components]: {np.mean(component_counts):.6f}")
    print(f"  P(connected): {connected_count/n_samples:.6f}")
    print(f"  E[chain length]: {np.mean(chain_lengths):.4f}")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    # Print analytic derivations first
    analytic_results()

    # Exact enumeration for small N
    results = {}
    for N in [2, 3, 4, 5]:
        results[N] = enumerate_all_2orders(N)

    # N=6: exact enumeration (720^2 = 518400 — feasible but skip Tutte/automorphisms)
    # We handle N=6 specially with reduced computations
    print(f"\n{'='*70}")
    print(f"N = 6: Exact enumeration (518,400 2-orders, reduced computations)")
    print(f"{'='*70}")

    perms6 = list(permutations(range(6)))
    n6 = len(perms6)
    total6 = n6 * n6

    max6 = []
    min6 = []
    cl6 = []
    of6 = []
    comp6 = []
    conn6 = 0
    ac2_6 = []
    chi6 = []
    aut6_sum = 0

    for ui, u in enumerate(perms6):
        if ui % 72 == 0:
            print(f"  Progress: {ui}/{n6}")
        for v in perms6:
            order = build_2order(u, v, 6)

            maximal = sum(1 for i in range(6) if not any(order[i][j] for j in range(6)))
            minimal = sum(1 for i in range(6) if not any(order[j][i] for j in range(6)))
            max6.append(maximal)
            min6.append(minimal)

            R = sum(1 for i in range(6) for j in range(6) if i != j and order[i][j])
            f_ord = R / 30
            of6.append(f_ord)

            cl = longest_chain(order, 6)
            cl6.append(cl)

            link = transitive_reduction(order, 6)
            edges = get_hasse_undirected_edges(link, 6)
            nc = count_connected_components(edges, 6)
            comp6.append(nc)
            if nc == 1:
                conn6 += 1

            sp = sum(1 for i in range(6) for j in range(i+1, 6) if not order[i][j] and not order[j][i])
            ac2_6.append(sp)

    H_6 = sum(Fraction(1, k) for k in range(1, 7))
    E_max6 = Fraction(sum(max6), total6)
    E_min6 = Fraction(sum(min6), total6)
    E_comp6 = Fraction(sum(comp6), total6)
    P_conn6 = Fraction(conn6, total6)
    E_ac2_6 = Fraction(sum(ac2_6), total6)

    print(f"\n  E[maximal] = {E_max6} = {float(E_max6):.6f} (H_6 = {H_6} = {float(H_6):.6f}, match={E_max6 == H_6})")
    print(f"  E[minimal] = {E_min6} = {float(E_min6):.6f} (match max? {E_max6 == E_min6})")
    print(f"  E[2-antichains] = {E_ac2_6} = {float(E_ac2_6):.6f} (predicted = {Fraction(30, 4)} = {30/4:.6f}, match={E_ac2_6 == Fraction(30, 4)})")
    print(f"  E[components] = {E_comp6} = {float(E_comp6):.6f}")
    print(f"  P(connected) = {P_conn6} = {float(P_conn6):.6f}")
    print(f"  Corr(chain, ordering) = {np.corrcoef(cl6, of6)[0,1]:.6f}")
    print(f"  Chain distribution: {dict(sorted(Counter(cl6).items()))}")

    results[6] = {
        'E_max': E_max6, 'E_min': E_min6,
        'E_components': E_comp6, 'P_connected': P_conn6,
        'E_ac2': E_ac2_6,
    }

    # Verify analytic formulas
    verify_analytic_maximal(results)
    verify_antichain_formula({N: r for N, r in results.items() if 'antichain_k' in r})

    # Monte Carlo for N=7,8
    for N in [7, 8]:
        monte_carlo_larger_N(N, n_samples=50000)

    # =================================================================
    # SUMMARY OF NEW EXACT RESULTS
    # =================================================================
    print("\n" + "="*70)
    print("SUMMARY OF NEW EXACT RESULTS FOR PAPER G")
    print("="*70)

    print("""
NEW THEOREM 1 (Expected maximal/minimal elements):
  E[number of maximal elements] = E[number of minimal elements] = H_N
  where H_N is the N-th harmonic number.

  Proof: Element at u-rank r is maximal iff it has the largest v-value
  among positions r, r+1, ..., N-1. Probability = 1/(N-r).
  Summing: E[maximal] = Sum_{r=0}^{N-1} 1/(N-r) = H_N.

NEW THEOREM 2 (Expected k-element antichains):
  E[number of k-element antichains] = C(N,k) / k!

  Special cases:
  - k=1: E = N (trivially)
  - k=2: E = N(N-1)/4 (spacelike pairs = half of all pairs)
  - k=3: E = N(N-1)(N-2)/36

  Proof: k elements form an antichain iff, sorted by u-rank, their
  v-values form the unique strictly decreasing permutation. P = 1/k!.

NEW EXACT VALUES (P(Hasse connected)):
  Exact P(connected) for N=2,3,4,5,6 computed by full enumeration.

NEW EXACT VALUES (Correlation chain-ordering):
  Exact Pearson correlation between longest chain and ordering fraction
  for N=2,3,4,5,6.

NEW EXACT VALUES (Chromatic number distribution):
  Full distribution of chromatic number of Hasse diagram for N=2,3,4,5.

NEW EXACT VALUES (Automorphism group size):
  E[|Aut(poset)|] and full distribution for N=2,3,4,5.

NEW EXACT VALUES (Tutte polynomial):
  Average Tutte polynomial of Hasse diagram for N=2,3,4,5.

NOTE on joint distribution:
  Joint distribution of (ordering fraction, link count) tabulated
  for N=4,5, showing the conditional E[links | ordering fraction].
""")

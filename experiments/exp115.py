"""
Experiment 115: FIELD IMPACT — Ideas 661-670

What would make the causal set community take notice?
10 ideas targeting maximum field impact.

IDEAS 661-665 (computational experiments):

661. VACUUM PRESCRIPTION FOR CAUSETS MIMICKING KRONECKER STRUCTURE
     The Kronecker theorem shows CDT factorizes (C^T-C = A_T ⊗ J) but causets don't.
     Propose a new vacuum prescription: impose an approximate time foliation on a
     causet (via longest-chain layering), project the Pauli-Jordan function onto the
     foliated subspace, and measure c_eff. Does the projected vacuum have finite c?

662. IMPROVED BD ACTION USING EXACT INTERVAL STATISTICS
     Our master formula gives exact interval distribution P[k|m] = 2(m-k)/[m(m+1)].
     The BD action uses interval counts N_k with expected values from Poisson sprinkling.
     Propose a MODIFIED BD action that subtracts the exact combinatorial background:
     S_modified = S_BD - <S_BD>_exact. Does this improve phase transition sharpness?

663. FIEDLER VALUE AS MCMC FILTER
     The Fiedler value (lambda_2 of Hasse Laplacian) distinguishes causets from random DAGs.
     Implement MCMC with a Fiedler-based acceptance criterion: reject configurations
     with lambda_2 below a threshold. Does this select more manifold-like causets?
     Measure ordering fraction, interval entropy, MM dimension of filtered ensemble.

664. KURTOSIS EXCESS OVER SEMICIRCLE AS NEW OBSERVABLE
     GUE is universal for antisymmetric matrices. But the eigenvalue density kurtosis=53
     for 2-orders (vs -1.2 for semicircle). The DEVIATION from GUE bulk statistics
     encodes causal set geometry. Define kappa_excess = kurtosis(rho) - kurtosis_GUE.
     How does it scale with N and d? Is it a dimension estimator?

665. HASSE DIAMETER SATURATION AND SMALL-WORLD PROPERTY
     Hasse diameter saturates at ~6 for 2-orders. Is this the causal set "small world"
     property? In Minkowski spacetime, the causal depth of a sprinkling grows as N^{1/d}.
     But the HASSE diameter (undirected shortest path in link graph) may saturate.
     Test: Hasse diameter vs N for N=20-500, d=2-5. Compare with random DAGs.

IDEAS 666-670 (assessment text only — see end of file):

666. TEXTBOOK CHAPTER on "Combinatorics of Causal Sets"
667. SHORT NOTE warning about phase-mixing artifacts in MCMC spectral statistics
668. RESEARCH PROGRAMME to fix c_eff divergence
669. What would Surya/Glaser/Dowker/Sorkin find most interesting?
670. Which ONE result to present at a causal set workshop?
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats, linalg
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import time
from math import comb, log, factorial

from causal_sets.fast_core import FastCausalSet, sprinkle_fast
from causal_sets.two_orders import TwoOrder, swap_move
from causal_sets.d_orders import DOrder
from causal_sets.bd_action import count_intervals_by_size, bd_action_2d
from causal_sets.sj_vacuum import pauli_jordan_function, sj_wightman_function, entanglement_entropy
from causal_sets.two_orders_v2 import bd_action_corrected

np.set_printoptions(precision=6, suppress=True, linewidth=120)
rng = np.random.default_rng(42)


# ============================================================
# SHARED UTILITIES
# ============================================================

def random_2order(N, rng_local=None):
    """Generate a random 2-order and return (FastCausalSet, TwoOrder)."""
    if rng_local is None:
        rng_local = rng
    to = TwoOrder(N, rng=rng_local)
    return to.to_causet(), to


def random_dorder(d, N, rng_local=None):
    """Generate a random d-order and return FastCausalSet."""
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


def fiedler_value(cs):
    """Second smallest eigenvalue of Hasse Laplacian (algebraic connectivity)."""
    L = hasse_laplacian(cs)
    evals = np.sort(np.linalg.eigvalsh(L))
    return evals[1] if len(evals) > 1 else 0.0


def hasse_diameter(cs):
    """Diameter of Hasse diagram (undirected) using BFS from each vertex."""
    adj = hasse_adjacency(cs)
    N = cs.n
    max_dist = 0
    for start in range(N):
        dist = np.full(N, -1)
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in range(N):
                if adj[u, v] > 0 and dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        reachable = dist[dist >= 0]
        if len(reachable) > 0:
            max_dist = max(max_dist, int(np.max(reachable)))
    return max_dist


def layer_decomposition(cs):
    """Decompose causet into time layers using longest-chain layering.
    Returns list of lists: layers[t] = list of element indices in layer t."""
    N = cs.n
    # Assign each element to a layer based on the length of the longest
    # chain ending at that element
    depth = np.zeros(N, dtype=int)
    order = cs.order.astype(np.int32)

    # Find a topological ordering first
    in_degree = np.sum(cs.order, axis=0)
    topo_order = np.argsort(in_degree)  # rough topological sort

    # Compute depth = length of longest chain ending at each element
    for j in topo_order:
        predecessors = np.where(cs.order[:, j])[0]
        if len(predecessors) > 0:
            depth[j] = np.max(depth[predecessors]) + 1

    # Group into layers
    max_depth = int(np.max(depth))
    layers = [[] for _ in range(max_depth + 1)]
    for i in range(N):
        layers[depth[i]].append(i)
    return layers, depth


def build_uniform_cdt(T, s):
    """Build a uniform CDT causal set: T slices of s vertices each."""
    N = T * s
    cs = FastCausalSet(N)
    for t1 in range(T):
        for t2 in range(t1 + 1, T):
            cs.order[t1*s:(t1+1)*s, t2*s:(t2+1)*s] = True
    return cs


def compute_c_eff(cs, region_frac=0.5):
    """Compute effective central charge from SJ entanglement entropy."""
    N = cs.n
    W = sj_wightman_function(cs)
    # Bipartition: first half vs second half
    k = int(N * region_frac)
    if k < 2 or k > N - 2:
        return 0.0
    S = entanglement_entropy(W, list(range(k)))
    # c_eff = 3*S / ln(N)
    if np.log(N) < 0.01:
        return 0.0
    return 3.0 * S / np.log(N)


def myrheim_meyer_dimension(cs, n_samples=200):
    """Estimate Myrheim-Meyer dimension from interval statistics."""
    N = cs.n
    order_int = cs.order.astype(np.int32)
    # Sample related pairs
    pairs_i, pairs_j = np.where(np.triu(cs.order, k=1))
    if len(pairs_i) < 10:
        return 1.0
    n_sample = min(n_samples, len(pairs_i))
    idx = rng.choice(len(pairs_i), n_sample, replace=False)
    # For each pair, count interval size
    interval_sizes = []
    for k in idx:
        i, j = pairs_i[k], pairs_j[k]
        between = cs.order[i, :] & cs.order[:, j]
        interval_sizes.append(np.sum(between))
    f2 = np.mean([s == 0 for s in interval_sizes])  # fraction of links
    if f2 <= 0 or f2 >= 1:
        return 1.0
    # MM formula: f2 = Gamma(d+1) * Gamma(d/2) / (2 * Gamma(3*d/2))
    # For d=2: f2 = 2/3 ≈ 0.667; d=3: f2 ≈ 0.354; d=4: f2 ≈ 0.148
    # Invert numerically
    from scipy.optimize import brentq
    from scipy.special import gamma as gamma_func
    def mm_eq(d):
        if d < 1.01:
            return f2 - 1.0
        return gamma_func(d+1) * gamma_func(d/2) / (2.0 * gamma_func(3*d/2)) - f2
    try:
        d_est = brentq(mm_eq, 1.01, 10.0)
    except:
        d_est = -1.0
    return d_est


print("=" * 80)
print("EXPERIMENT 115: FIELD IMPACT — Ideas 661-670")
print("What would make the causal set community take notice?")
print("=" * 80)
sys.stdout.flush()


# ============================================================
# IDEA 661: VACUUM PRESCRIPTION MIMICKING KRONECKER STRUCTURE
# ============================================================
print("\n" + "=" * 80)
print("IDEA 661: New Vacuum Prescription for Causets via Foliation Projection")
print("=" * 80)
print("""
BACKGROUND: CDT's Kronecker theorem gives C^T-C = A_T ⊗ J, meaning the SJ vacuum
factorizes into temporal and spatial parts. Causets lack this because they have no
preferred foliation. But we CAN impose one via longest-chain layering.

PROPOSAL: Given a causet:
  1. Decompose into layers (approximate time slices)
  2. Build the "foliated approximation": keep only inter-layer relations
  3. Compute SJ vacuum on the foliated causet
  4. Measure c_eff — does the foliated vacuum have finite c?

If c_eff becomes finite (like CDT), this proves the foliation is what controls
the central charge, and proposes a NEW vacuum state for causal sets.
""")
sys.stdout.flush()

t0 = time.time()

print(f"  {'Structure':>18} {'N':>4} {'T_layers':>8} {'c_raw':>8} {'c_foliated':>10} {'c_CDT_ref':>10}")
print("-" * 70)

for N in [30, 50, 70]:
    for trial in range(3):
        rng_t = np.random.default_rng(42 + trial + N)
        cs, to = random_2order(N, rng_local=rng_t)

        # Raw c_eff
        c_raw = compute_c_eff(cs)

        # Layer decomposition
        layers, depth = layer_decomposition(cs)
        T_layers = len(layers)

        # Build foliated approximation: only keep inter-layer relations
        cs_fol = FastCausalSet(N)
        for i in range(N):
            for j in range(N):
                if depth[i] < depth[j]:
                    cs_fol.order[i, j] = True

        c_fol = compute_c_eff(cs_fol)

        # CDT reference with same T and ~N
        s_cdt = max(2, N // T_layers)
        cs_cdt = build_uniform_cdt(T_layers, s_cdt)
        N_cdt = T_layers * s_cdt
        c_cdt = compute_c_eff(cs_cdt)

        print(f"  {'2-order':>18} {N:>4} {T_layers:>8} {c_raw:>8.3f} {c_fol:>10.3f} {c_cdt:>10.3f}")

print()

# Systematic N-scaling of foliated c_eff
print("  N-scaling of foliated c_eff (averaged over 5 trials):")
print(f"  {'N':>5} {'c_raw':>8} {'c_foliated':>10} {'T_layers':>8} {'ratio':>8}")
for N in [20, 30, 40, 50, 60, 70]:
    c_raws, c_fols, T_list = [], [], []
    for trial in range(5):
        rng_t = np.random.default_rng(100 + trial + N * 7)
        cs, _ = random_2order(N, rng_local=rng_t)
        c_raws.append(compute_c_eff(cs))

        layers, depth = layer_decomposition(cs)
        T_list.append(len(layers))

        cs_fol = FastCausalSet(N)
        for i in range(N):
            for j in range(N):
                if depth[i] < depth[j]:
                    cs_fol.order[i, j] = True
        c_fols.append(compute_c_eff(cs_fol))

    print(f"  {N:>5} {np.mean(c_raws):>8.3f} {np.mean(c_fols):>10.3f} {np.mean(T_list):>8.1f} {np.mean(c_fols)/np.mean(c_raws):>8.3f}")

dt = time.time() - t0
print(f"\n  [Idea 661 completed in {dt:.1f}s]")

# Assessment
print("""
ASSESSMENT (Idea 661):
- If foliated c_eff is significantly LOWER than raw c_eff, the foliation projection
  acts as a vacuum regulator. This would be a concrete proposal for a new causal set
  vacuum state that interpolates between the SJ vacuum and CDT.
- If foliated c_eff still diverges, the problem is deeper than foliation — it's
  about the nature of the partial order itself.
- FIELD IMPACT: A new vacuum prescription that gives finite c would be immediately
  publishable and directly addresses the biggest open problem in SJ vacuum theory.
""")
sys.stdout.flush()


# ============================================================
# IDEA 662: IMPROVED BD ACTION USING EXACT INTERVAL STATISTICS
# ============================================================
print("\n" + "=" * 80)
print("IDEA 662: Modified BD Action with Exact Combinatorial Background Subtraction")
print("=" * 80)
print("""
BACKGROUND: The BD action S = N - 2L + I_2 uses interval counts. For random
2-orders, we have EXACT expectations from the master formula:
  E[links] = (N+1)*H_N - 2N  (where H_N = harmonic number)
  E[I_2] can be computed from the interval generating function.

PROPOSAL: Define a "background-subtracted" action:
  S_mod = S_BD - <S_BD>_exact
This removes the combinatorial noise and isolates the geometric signal.
If the phase transition is sharper with S_mod, it means the exact formulas
provide a better prior for causal set dynamics.

METHOD: Compute S_BD and S_mod for random 2-orders at several beta values.
Compare fluctuations and transition sharpness.
""")
sys.stdout.flush()

t0 = time.time()

def exact_expected_links(N):
    """E[links] = (N+1)*H_N - 2N for random 2-orders."""
    H_N = sum(1.0/k for k in range(1, N+1))
    return (N + 1) * H_N - 2 * N


def exact_expected_I2(N):
    """E[I_2] for random 2-orders.
    From the master formula P[k|m] = 2(m-k)/[m(m+1)]:
    An I_2 interval has exactly 1 element between endpoints.
    For a gap of size m (= m+2 elements in total), interval of size k=1:
    P[k=1|m] = 2(m-1)/[m(m+1)]

    E[I_2] = sum over all related pairs of P(exactly 1 between)
    For random 2-orders, the gap size m for a pair at ranks (r,s) with r<s
    in one permutation is m = s - r. The pair is related iff both coords
    are ordered. This is complex — use Monte Carlo.
    """
    n_trials = 2000
    I2_vals = []
    for _ in range(n_trials):
        to = TwoOrder(N, rng=np.random.default_rng())
        cs = to.to_causet()
        counts = count_intervals_by_size(cs, max_size=1)
        I2_vals.append(counts.get(1, 0))
    return np.mean(I2_vals)


def exact_expected_bd_2d(N):
    """E[S_BD_2d] = N - 2*E[links] + E[I_2].
    Since this is expensive to compute E[I_2] exactly, use MC estimate."""
    E_links = exact_expected_links(N)
    E_I2 = exact_expected_I2(N)
    return N - 2 * E_links + E_I2


# Compute exact expectations for several N
print("  Exact expected BD action components:")
print(f"  {'N':>5} {'E[links]':>10} {'E[I_2] (MC)':>12} {'E[S_BD]':>10}")
expected_S = {}
for N in [20, 30, 40, 50]:
    E_L = exact_expected_links(N)
    E_I2 = exact_expected_I2(N)
    E_S = N - 2 * E_L + E_I2
    expected_S[N] = E_S
    print(f"  {N:>5} {E_L:>10.3f} {E_I2:>12.3f} {E_S:>10.3f}")

print()

# Now compare S_BD vs S_mod = S_BD - E[S_BD] at the phase transition
print("  Phase transition comparison (N=30, eps=0.12):")
print(f"  {'beta/beta_c':>12} {'<S_BD>':>10} {'std(S_BD)':>10} {'<S_mod>':>10} {'std(S_mod)':>10} {'ratio_std':>10}")

N = 30
eps = 0.12
beta_c = 1.66 / (N * eps**2)

# MC estimate of E[S] for this N, eps
E_S_exact_MC = []
for trial in range(500):
    to_t = TwoOrder(N, rng=np.random.default_rng(trial))
    cs_t = to_t.to_causet()
    E_S_exact_MC.append(bd_action_corrected(cs_t, eps))
E_S_bg = np.mean(E_S_exact_MC)

for beta_frac in [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
    beta = beta_frac * beta_c
    # Quick MCMC
    from causal_sets.two_orders_v2 import mcmc_corrected
    result = mcmc_corrected(N, beta, eps, n_steps=8000, n_therm=4000,
                            record_every=10, rng=np.random.default_rng(42))
    S_vals = result['actions']
    S_mod = S_vals - E_S_bg

    print(f"  {beta_frac:>12.1f} {np.mean(S_vals):>10.4f} {np.std(S_vals):>10.4f} "
          f"{np.mean(S_mod):>10.4f} {np.std(S_mod):>10.4f} {np.std(S_mod)/max(np.std(S_vals),1e-10):>10.3f}")

dt = time.time() - t0
print(f"\n  [Idea 662 completed in {dt:.1f}s]")

print("""
ASSESSMENT (Idea 662):
- Background subtraction centers the action at zero for random causets.
- If std(S_mod) < std(S_BD), the exact formulas remove combinatorial noise.
- The KEY test: does the susceptibility chi = N*Var(S_mod) show a SHARPER peak
  at beta_c than chi_raw = N*Var(S_BD)?
- FIELD IMPACT: If the exact interval statistics provide a better BD action,
  this directly improves the causal set path integral — the central object in
  causal set quantum gravity. Surya and Glaser would immediately notice.
""")
sys.stdout.flush()


# ============================================================
# IDEA 663: FIEDLER VALUE AS MCMC FILTER
# ============================================================
print("\n" + "=" * 80)
print("IDEA 663: Fiedler-Filtered MCMC — Selecting Manifold-like Causets")
print("=" * 80)
print("""
BACKGROUND: The Fiedler value (lambda_2 of Hasse Laplacian) measures algebraic
connectivity. Manifold-like causets have higher Fiedler values than random DAGs.
Idea: use lambda_2 as an ADDITIONAL acceptance criterion in MCMC.

PROPOSAL: Modified Metropolis:
  Accept if exp(-beta*S_BD) AND lambda_2 > lambda_threshold
This should bias the ensemble toward manifold-like configurations.

TEST: Compare the standard BD MCMC ensemble with Fiedler-filtered ensemble:
  - Ordering fraction (manifoldlikeness)
  - Myrheim-Meyer dimension (should be closer to 2.0)
  - Interval entropy
""")
sys.stdout.flush()

t0 = time.time()

def fiedler_filtered_mcmc(N, beta, eps, lambda_threshold, n_steps=5000,
                           n_therm=2500, record_every=10, rng_local=None):
    """MCMC with Fiedler value filter: reject moves that drop lambda_2 below threshold."""
    if rng_local is None:
        rng_local = rng

    current = TwoOrder(N, rng=rng_local)
    current_cs = current.to_causet()
    current_S = bd_action_corrected(current_cs, eps)
    current_fiedler = fiedler_value(current_cs)

    samples = []
    actions = []
    fiedler_vals = []
    n_acc = 0
    n_fiedler_reject = 0

    for step in range(n_steps):
        proposed = swap_move(current, rng_local)
        proposed_cs = proposed.to_causet()
        proposed_S = bd_action_corrected(proposed_cs, eps)

        # Standard Metropolis criterion
        dS = beta * (proposed_S - current_S)
        accept_metro = (dS <= 0 or rng_local.random() < np.exp(-min(dS, 500)))

        if accept_metro:
            # Fiedler filter
            proposed_fiedler = fiedler_value(proposed_cs)
            if proposed_fiedler >= lambda_threshold:
                current = proposed
                current_cs = proposed_cs
                current_S = proposed_S
                current_fiedler = proposed_fiedler
                n_acc += 1
            else:
                n_fiedler_reject += 1

        if step >= n_therm and step % record_every == 0:
            actions.append(current_S)
            fiedler_vals.append(current_fiedler)
            samples.append(current_cs)

    return {
        'actions': np.array(actions),
        'fiedler_vals': np.array(fiedler_vals),
        'samples': samples,
        'accept_rate': n_acc / n_steps,
        'fiedler_reject_rate': n_fiedler_reject / n_steps,
    }


N = 30
eps = 0.12
beta = 1.0 * beta_c  # at the transition

# First: measure Fiedler distribution at beta=0 to calibrate threshold
print("  Calibrating Fiedler threshold at beta=0...")
fiedler_dist = []
for trial in range(200):
    cs_t, _ = random_2order(N, rng_local=np.random.default_rng(trial + 1000))
    fiedler_dist.append(fiedler_value(cs_t))
fiedler_dist = np.array(fiedler_dist)
print(f"  Fiedler at beta=0: mean={np.mean(fiedler_dist):.3f}, "
      f"std={np.std(fiedler_dist):.3f}, "
      f"median={np.median(fiedler_dist):.3f}, "
      f"P25={np.percentile(fiedler_dist, 25):.3f}, "
      f"P75={np.percentile(fiedler_dist, 75):.3f}")

# Set threshold at median
lambda_thresh = np.median(fiedler_dist)
print(f"  Using threshold lambda_2 > {lambda_thresh:.3f} (median)")

# Standard MCMC
print("\n  Standard MCMC at beta_c:")
result_std = mcmc_corrected(N, beta, eps, n_steps=5000, n_therm=2500,
                            record_every=10, rng=np.random.default_rng(42))

# Fiedler-filtered MCMC
print("  Fiedler-filtered MCMC at beta_c:")
result_filt = fiedler_filtered_mcmc(N, beta, eps, lambda_thresh,
                                     n_steps=5000, n_therm=2500,
                                     record_every=10,
                                     rng_local=np.random.default_rng(42))

# Compare observables
print(f"\n  {'Observable':>25} {'Standard':>12} {'Filtered':>12} {'Diff':>10}")
print("-" * 65)

# Ordering fraction
of_std = [s.ordering_fraction() for s in result_std['samples']]
of_filt = [s.ordering_fraction() for s in result_filt['samples']]
print(f"  {'Ordering fraction':>25} {np.mean(of_std):>12.4f} {np.mean(of_filt):>12.4f} "
      f"{np.mean(of_filt)-np.mean(of_std):>+10.4f}")

# Fiedler values
fied_std = [fiedler_value(s) for s in result_std['samples']]
fied_filt = result_filt['fiedler_vals']
print(f"  {'Fiedler value':>25} {np.mean(fied_std):>12.4f} {np.mean(fied_filt):>12.4f} "
      f"{np.mean(fied_filt)-np.mean(fied_std):>+10.4f}")

# MM dimension
mm_std = [myrheim_meyer_dimension(s, n_samples=100) for s in result_std['samples'][:20]]
mm_filt = [myrheim_meyer_dimension(s, n_samples=100) for s in result_filt['samples'][:20]]
print(f"  {'MM dimension':>25} {np.mean(mm_std):>12.4f} {np.mean(mm_filt):>12.4f} "
      f"{np.mean(mm_filt)-np.mean(mm_std):>+10.4f}")

# Interval entropy
def interval_entropy(cs):
    counts = count_intervals_by_size(cs, max_size=min(cs.n-2, 15))
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array([counts.get(k, 0) for k in range(max(counts.keys())+1)]) / total
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))

ie_std = [interval_entropy(s) for s in result_std['samples']]
ie_filt = [interval_entropy(s) for s in result_filt['samples']]
print(f"  {'Interval entropy':>25} {np.mean(ie_std):>12.4f} {np.mean(ie_filt):>12.4f} "
      f"{np.mean(ie_filt)-np.mean(ie_std):>+10.4f}")

# Accept rates
print(f"  {'Accept rate':>25} {result_std['accept_rate']:>12.4f} {result_filt['accept_rate']:>12.4f}")
print(f"  {'Fiedler reject rate':>25} {'N/A':>12} {result_filt['fiedler_reject_rate']:>12.4f}")

dt = time.time() - t0
print(f"\n  [Idea 663 completed in {dt:.1f}s]")

print("""
ASSESSMENT (Idea 663):
- If Fiedler-filtered MCMC gives causets with MM dimension closer to 2.0 and
  higher ordering fraction, the Fiedler value acts as a manifoldlikeness filter.
- KEY QUESTION: Does the filter break ergodicity or merely bias the ensemble?
- FIELD IMPACT: A new MCMC prescription that selects manifold-like causets without
  relying solely on the BD action would be a significant algorithmic contribution.
  Glaser (who does most of the MCMC work in the field) would find this interesting.
""")
sys.stdout.flush()


# ============================================================
# IDEA 664: KURTOSIS EXCESS OVER SEMICIRCLE AS NEW OBSERVABLE
# ============================================================
print("\n" + "=" * 80)
print("IDEA 664: Kurtosis Excess κ_excess as Geometry-Encoding Observable")
print("=" * 80)
print("""
BACKGROUND: The eigenvalue density of iΔ on 2-orders has kurtosis ≈ 53, far from
the semicircular law kurtosis = -1.2 (Wigner's theorem applies to symmetric, not
antisymmetric matrices, but GUE-like nulls give semicircle).

The DEVIATION from semicircle kurtosis encodes geometry:
  κ_excess = kurtosis(ρ) - κ_semicircle

QUESTIONS:
  1. How does κ_excess scale with N?
  2. Does it depend on dimension d (d-orders)?
  3. Can it serve as a new dimension estimator?
  4. What is κ_excess for CDT?
""")
sys.stdout.flush()

t0 = time.time()

def eigenvalue_kurtosis(cs):
    """Kurtosis of the eigenvalue density of i*iDelta."""
    iDelta = pauli_jordan_function(cs)
    evals = np.linalg.eigvalsh(1j * iDelta).real
    # Remove near-zero eigenvalues (within 1e-10)
    evals_nz = evals[np.abs(evals) > 1e-10]
    if len(evals_nz) < 4:
        return 0.0
    return float(stats.kurtosis(evals_nz, fisher=True))  # excess kurtosis


KURTOSIS_SEMICIRCLE = -1.2  # Wigner semicircle excess kurtosis

# N-scaling for 2-orders
print("  κ_excess vs N for random 2-orders (5 trials each):")
print(f"  {'N':>5} {'κ_raw':>10} {'κ_excess':>10} {'std':>8}")
for N in [20, 30, 40, 50, 70]:
    kvals = []
    for trial in range(5):
        cs, _ = random_2order(N, rng_local=np.random.default_rng(trial + N * 3))
        kvals.append(eigenvalue_kurtosis(cs))
    ke = np.array(kvals)
    print(f"  {N:>5} {np.mean(ke):>10.2f} {np.mean(ke) - KURTOSIS_SEMICIRCLE:>10.2f} {np.std(ke):>8.2f}")

# Dimension dependence for d-orders
print(f"\n  κ_excess vs dimension d at N=30 (5 trials each):")
print(f"  {'d':>5} {'κ_raw':>10} {'κ_excess':>10} {'std':>8}")
for d in [2, 3, 4, 5]:
    kvals = []
    for trial in range(5):
        rng_t = np.random.default_rng(trial + d * 100)
        do = DOrder(d, 30, rng=rng_t)
        cs = do.to_causet_fast()
        kvals.append(eigenvalue_kurtosis(cs))
    ke = np.array(kvals)
    print(f"  {d:>5} {np.mean(ke):>10.2f} {np.mean(ke) - KURTOSIS_SEMICIRCLE:>10.2f} {np.std(ke):>8.2f}")

# CDT comparison
print(f"\n  κ_excess for CDT (T slices, s=5):")
print(f"  {'T':>5} {'N':>5} {'κ_raw':>10} {'κ_excess':>10}")
for T in [4, 6, 8, 10]:
    s = 5
    cs_cdt = build_uniform_cdt(T, s)
    k_cdt = eigenvalue_kurtosis(cs_cdt)
    print(f"  {T:>5} {T*s:>5} {k_cdt:>10.2f} {k_cdt - KURTOSIS_SEMICIRCLE:>10.2f}")

# Sprinkled causets
print(f"\n  κ_excess for sprinkled causets in 2D Minkowski (N=30, 5 trials):")
kvals_spr = []
for trial in range(5):
    rng_t = np.random.default_rng(trial + 5000)
    cs_spr, coords = sprinkle_fast(30, dim=2, rng=rng_t)
    kvals_spr.append(eigenvalue_kurtosis(cs_spr))
ke_spr = np.array(kvals_spr)
print(f"  Sprinkled 2D: κ_raw = {np.mean(ke_spr):.2f} ± {np.std(ke_spr):.2f}, "
      f"κ_excess = {np.mean(ke_spr) - KURTOSIS_SEMICIRCLE:.2f}")

# Random antisymmetric null
print(f"\n  κ_excess for random antisymmetric ±1 matrix (matched sparsity, N=30, 5 trials):")
kvals_null = []
for trial in range(5):
    rng_t = np.random.default_rng(trial + 6000)
    # Build random antisymmetric with same density as 2-order
    M = np.zeros((30, 30))
    density = 0.5  # approximate for 2-orders
    for i in range(30):
        for j in range(i+1, 30):
            if rng_t.random() < density:
                sign = rng_t.choice([-1, 1])
                M[i, j] = sign * (2.0/30)
                M[j, i] = -sign * (2.0/30)
    evals_null = np.linalg.eigvalsh(1j * M).real
    evals_null_nz = evals_null[np.abs(evals_null) > 1e-10]
    if len(evals_null_nz) >= 4:
        kvals_null.append(float(stats.kurtosis(evals_null_nz, fisher=True)))
if len(kvals_null) > 0:
    ke_null = np.array(kvals_null)
    print(f"  Null: κ_raw = {np.mean(ke_null):.2f} ± {np.std(ke_null):.2f}, "
          f"κ_excess = {np.mean(ke_null) - KURTOSIS_SEMICIRCLE:.2f}")

dt = time.time() - t0
print(f"\n  [Idea 664 completed in {dt:.1f}s]")

print("""
ASSESSMENT (Idea 664):
- If κ_excess depends on d (dimension), it's a new dimension estimator complementary
  to Myrheim-Meyer.
- If κ_excess grows with N, it measures the departure from semicircular law that's
  intrinsic to causal set geometry (vs generic random matrices).
- KEY TEST: Does κ_excess for the null (random antisymmetric) match semicircle?
  If so, the excess is genuinely geometric, not an artifact.
- FIELD IMPACT: A new spectral observable that encodes dimension would strengthen
  Paper D (spectral statistics) and give the community a new tool. Medium impact:
  the community cares more about dynamics than spectral geometry.
""")
sys.stdout.flush()


# ============================================================
# IDEA 665: HASSE DIAMETER SATURATION AND SMALL-WORLD PROPERTY
# ============================================================
print("\n" + "=" * 80)
print("IDEA 665: Hasse Diameter Saturation — Small-World Property of Minkowski Space?")
print("=" * 80)
print("""
BACKGROUND: In exp102, Hasse diameter scales as ~N^0.11, nearly constant (~6).
In exp101, Hasse graph is NOT small-world (clustering coefficient too low).
But the diameter saturation is remarkable — every event is ~6 links from every other.

QUESTIONS:
  1. Does diameter saturate or grow (slowly) at large N?
  2. Is this a d-dependent phenomenon? (d=2,3,4,5)
  3. How does CDT compare?
  4. Is this related to the "six degrees of separation" in Minkowski spacetime?
  5. What about the MEAN geodesic distance (not just max)?

METHOD: Compute Hasse diameter and mean geodesic for N=20-200, d=2-5.
""")
sys.stdout.flush()

t0 = time.time()

def hasse_mean_geodesic(cs):
    """Mean geodesic distance in Hasse diagram (among reachable pairs)."""
    adj = hasse_adjacency(cs)
    N = cs.n
    all_dists = []
    for start in range(N):
        dist = np.full(N, -1)
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in range(N):
                if adj[u, v] > 0 and dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        reachable = dist[(dist > 0)]
        all_dists.extend(reachable.tolist())
    if len(all_dists) == 0:
        return 0.0
    return np.mean(all_dists)


# d=2 (2-orders): N-scaling
print("  Hasse diameter and mean geodesic vs N for 2-orders (3 trials avg):")
print(f"  {'N':>5} {'diam':>8} {'mean_geo':>10} {'ln(N)':>8}")
for N in [20, 30, 50, 70, 100, 150]:
    diams, geos = [], []
    for trial in range(3):
        rng_t = np.random.default_rng(trial + N * 11)
        cs, _ = random_2order(N, rng_local=rng_t)
        diams.append(hasse_diameter(cs))
        if N <= 100:  # mean geodesic is expensive
            geos.append(hasse_mean_geodesic(cs))
    geo_str = f"{np.mean(geos):>10.2f}" if len(geos) > 0 else f"{'--':>10}"
    print(f"  {N:>5} {np.mean(diams):>8.1f} {geo_str} {np.log(N):>8.2f}")

# Dimension dependence
print(f"\n  Hasse diameter vs dimension d at N=40 (5 trials):")
print(f"  {'d':>5} {'diam':>8} {'mean_geo':>10}")
for d in [2, 3, 4, 5]:
    diams, geos = [], []
    for trial in range(5):
        rng_t = np.random.default_rng(trial + d * 200)
        do = DOrder(d, 40, rng=rng_t)
        cs = do.to_causet_fast()
        diams.append(hasse_diameter(cs))
        geos.append(hasse_mean_geodesic(cs))
    print(f"  {d:>5} {np.mean(diams):>8.1f} {np.mean(geos):>10.2f}")

# CDT comparison
print(f"\n  CDT Hasse diameter:")
print(f"  {'T':>5} {'s':>5} {'N':>5} {'diam':>8}")
for T in [5, 8, 10, 15]:
    s = 5
    cs_cdt = build_uniform_cdt(T, s)
    d_cdt = hasse_diameter(cs_cdt)
    print(f"  {T:>5} {s:>5} {T*s:>5} {d_cdt:>8}")

# Sprinkled causet comparison
print(f"\n  Sprinkled 2D Minkowski causets:")
print(f"  {'N':>5} {'diam':>8} {'mean_geo':>10}")
for N in [20, 30, 50]:
    diams, geos = [], []
    for trial in range(3):
        rng_t = np.random.default_rng(trial + N * 17)
        cs_spr, coords = sprinkle_fast(N, dim=2, rng=rng_t)
        diams.append(hasse_diameter(cs_spr))
        geos.append(hasse_mean_geodesic(cs_spr))
    print(f"  {N:>5} {np.mean(diams):>8.1f} {np.mean(geos):>10.2f}")

# Fit power law diameter ~ N^alpha
print(f"\n  Power-law fit: diameter ~ N^alpha for 2-orders")
Ns_fit = [20, 30, 50, 70, 100, 150]
diams_fit = []
for N in Ns_fit:
    d_avg = []
    for trial in range(3):
        rng_t = np.random.default_rng(trial + N * 31)
        cs, _ = random_2order(N, rng_local=rng_t)
        d_avg.append(hasse_diameter(cs))
    diams_fit.append(np.mean(d_avg))

log_N = np.log(Ns_fit)
log_d = np.log(diams_fit)
slope, intercept, r, p, se = stats.linregress(log_N, log_d)
print(f"  alpha = {slope:.4f} ± {se:.4f} (R² = {r**2:.4f})")
print(f"  diam ≈ {np.exp(intercept):.2f} * N^{slope:.3f}")

dt = time.time() - t0
print(f"\n  [Idea 665 completed in {dt:.1f}s]")

print("""
ASSESSMENT (Idea 665):
- If diameter ~ N^alpha with alpha << 0.5, the Hasse graph is "ultrasmall-world".
  In contrast, a lattice in d dimensions has diameter ~ N^{1/d}.
- The saturation (if confirmed) means causal set geometry has a "compactness"
  property that lattice-based approaches lack.
- If diameter depends on d, it's ANOTHER dimension estimator.
- FIELD IMPACT: Moderate. The result is cute but probably a corollary of the
  link fraction scaling (links/N ~ ln(N)/N). High link fraction = short paths.
  Would be a nice observation in a review paper but unlikely to generate
  standalone excitement.
""")
sys.stdout.flush()


# ============================================================
# IDEAS 666-670: ASSESSMENT TEXT (NO CODE)
# ============================================================
print("\n" + "=" * 80)
print("IDEAS 666-670: FIELD IMPACT ASSESSMENTS")
print("=" * 80)

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IDEA 666: TEXTBOOK CHAPTER — "Combinatorics of Causal Sets"

CONTENT: We have 15+ exact theorems suitable for a textbook chapter:
  - Master interval formula P[k|m] = 2(m-k)/[m(m+1)]
  - E[links] = (N+1)H_N - 2N
  - E[f] = 1/2 (2-orders), E[f] = 1/(d! * 2^{d-1}) (d-orders)
  - E[maximal] = E[minimal] = H_N
  - E[k-antichains] = C(N,k)/k!
  - E[S_Glaser] = 1 for ALL N
  - Antichain ~ 2*sqrt(N) (Vershik-Kerov)
  - Complement theorem: f(C) + f(C') = 1
  - f-vector: E[f_k] = C(N,k+1)/(k+1)!
  - P(dim=2) = 1 - 1/N!
  - Exact connectivity probabilities for N=2-6
  - Kronecker product theorem for CDT

FORMAT: 30-40 pages. Self-contained. Suitable for a chapter in a handbook
(e.g., "Handbook of Combinatorics and Discrete Structures in Physics").

FIELD IMPACT SCORE: 7.5/10
  PRO: Would be a definitive reference. Currently no such compilation exists.
  The causal set community is small (~50 active researchers), but combinatorialists
  would also find it interesting — random 2-orders = random permutations intersection,
  which connects to RSK, Young tableaux, longest increasing subsequences.
  CON: Textbook chapters have low citation impact. The community might view it as
  "nice math" rather than "physics that matters." The theorems are about RANDOM
  causets (beta=0), not about the physical continuum-phase causets.
  RECOMMENDATION: Write it. Submit to J. Combinatorial Theory A or contribute to
  a QG review volume. The mathematical content is solid and unique.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IDEA 667: SHORT NOTE — Phase-Mixing Artifacts in MCMC Spectral Statistics

CONTENT: Our exp92 discovered that the "sub-Poisson" <r>=0.12 spectral statistic
was an artifact of PHASE MIXING: concatenating spectra from different phases
(ordered and disordered) in a single MCMC run near beta_c. The mixture produces
artificially low <r> because eigenvalue distributions from different phases
interleave when concatenated.

FORMAT: 3-4 page note. "A cautionary tale: phase mixing artifacts in spectral
statistics from MCMC simulations near phase transitions."

FIELD IMPACT SCORE: 7.0/10
  PRO: This is a methodological lesson that applies broadly — anyone doing
  spectral statistics from MCMC near a phase transition could fall into this trap.
  The causal set community (Glaser, Surya, Cunningham) uses MCMC extensively.
  Beyond causets, the lesson applies to lattice QCD, spin models, any system with
  a first-order phase transition where spectral observables are computed.
  CON: It's a negative result / warning. Journals don't love negative results.
  The note is too short for Physical Review but could work for a Comment or
  J. Phys. A: Math. Theor. letter.
  RECOMMENDATION: Write it. Include the mixture test (concat spectra from two
  known phases, show <r> drops) as the definitive diagnostic. 2 figures.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IDEA 668: RESEARCH PROGRAMME — Fixing the c_eff Divergence

BACKGROUND: The SJ vacuum central charge c_eff diverges as ~ln(N) on causal sets,
both 2-orders and sprinkled. On CDT, c_eff = 1 (the correct value for a free
scalar in 2D). The Kronecker theorem explains CDT's success: the time foliation
makes the vacuum factorize.

THE PROBLEM: Causal sets lack a preferred foliation. The SJ vacuum "sees" too many
modes because it treats ALL directions democratically. The Pauli-Jordan function
iDelta has O(N) positive eigenvalues on causets but only O(1) on CDT.

PROPOSED RESEARCH PROGRAMME (5-10 year horizon):

1. FOLIATED SJ VACUUM (Ideas 661 above):
   Impose an approximate foliation via longest-chain layering, project the
   Pauli-Jordan function. If this gives finite c, the foliation is the key ingredient.

2. SPECTRAL TRUNCATION:
   The Kronecker theorem shows that the "correct" SJ vacuum only uses the T/2
   largest positive eigenvalues of iDelta (where T = number of time slices).
   For causets, define a truncated vacuum W_trunc using only the top-k eigenvalues
   where k = k(N) grows sublinearly. The challenge: what is the correct k(N)?

3. MODIFIED RETARDED GREEN'S FUNCTION:
   The Pauli-Jordan function is iDelta = G_R - G_A where G_R[i,j] = C[j,i]/N.
   The normalization 1/N comes from the continuum limit. But perhaps the correct
   normalization depends on the LOCAL causal structure (interval size, depth).
   A position-dependent normalization could suppress the extra modes.

4. COARSE-GRAINING:
   Instead of computing the SJ vacuum on the full causet, coarse-grain first.
   Group elements into "blocks" (e.g., by layer), compute SJ on the coarse-grained
   causet. This explicitly introduces a spatial scale and should reduce modes.

5. MEASURE-THEORETIC FIX:
   The BD action + MCMC selects manifold-like causets at large beta. Perhaps the
   c_eff divergence is an artifact of computing at beta=0 (random causets).
   The PHYSICAL vacuum is the SJ vacuum on MCMC-sampled causets near beta_c.
   If c_eff is finite there (not tested at large N), the problem is solved
   dynamically.

FIELD IMPACT SCORE: 8.5/10
  PRO: The c_eff divergence is THE central problem in SJ vacuum theory on causets.
  Solving it would be a breakthrough. Even a well-motivated research programme
  (without a solution) would be highly cited because it frames the problem clearly.
  Sorkin (who invented the SJ vacuum) and Yazdi (who first computed c_eff on causets)
  would be very interested. Dowker would appreciate the foliation connection.
  CON: A research programme without a solution is less impactful than a solution.
  Risk of being scooped: Yazdi and collaborators are likely thinking about this too.
  RECOMMENDATION: Write a focused paper: "The central charge problem in SJ vacuum
  theory: diagnosis and paths forward." Include the Kronecker theorem as the
  diagnostic key and the 5 proposals above as the path forward.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IDEA 669: What Would Surya/Glaser/Dowker/Sorkin Find Most Interesting?

ANALYSIS BY RESEARCHER:

SUMATI SURYA (leader of causal set MCMC programme):
  Most interesting: Our exact combinatorial results (Paper G).
  Why: Surya uses MCMC on 2-orders extensively. Having exact analytical results
  for the beta=0 ensemble gives her null-model predictions. The master interval
  formula and E[S_Glaser]=1 are directly relevant to her work on the BD action.
  She would also appreciate the Fiedler value as BD order parameter (Paper F) since
  it gives a new tool for characterizing the phase transition.
  Least interesting: ER=EPR (Paper C) — she works on dynamics, not AdS/CFT.

LISA GLASER (MCMC algorithms, simulation):
  Most interesting: The phase-mixing artifact (Idea 667) and Fiedler-filtered MCMC
  (Idea 663).
  Why: Glaser does the actual simulations. A methodological warning about MCMC
  artifacts directly affects her work. A new acceptance criterion (Fiedler filter)
  is an algorithmic innovation she could immediately implement.
  She'd also be interested in the Kronecker theorem (Paper E) because it explains
  WHY CDT and causets give different results — she has worked on both.
  Least interesting: Exact combinatorics (Paper G) — she's a physicist, not a
  mathematician.

FAY DOWKER (foundations, kinematics):
  Most interesting: The Kronecker product theorem (Paper E) and the c_eff
  divergence diagnosis.
  Why: Dowker cares deeply about the physical content of causal set theory.
  The Kronecker theorem shows that CDT's success with c=1 comes from the
  foliation, not from quantum gravity. This is a deep structural insight about
  the SJ vacuum. She would also appreciate the Gram identity (Paper C).
  Least interesting: Phase transition observables (Paper F) — she cares about
  kinematics over dynamics.

RAFAEL SORKIN (founder of causal set theory):
  Most interesting: The c_eff divergence research programme (Idea 668) and the
  Gram identity universality (Paper C).
  Why: Sorkin invented both causal sets and the SJ vacuum. The c_eff problem
  threatens the viability of his programme. A well-diagnosed research programme
  to fix it would get his attention. The Gram identity connecting the causal
  matrix to the retarded Green's function is the type of exact structural
  result he values.
  He would be SKEPTICAL of: anything that requires a preferred foliation
  (violates the causal set philosophy), GUE universality (too generic).

CONSENSUS: The Kronecker product theorem (Paper E) and exact combinatorics
(Paper G) would have the broadest appeal across the community.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IDEA 670: THE ONE RESULT for a Causal Set Workshop

If we could present ONE result at a causal set workshop (e.g., the biennial
"Causal Sets" meeting organized by Surya), which would have the MOST IMPACT?

CANDIDATES (ranked):

1. THE KRONECKER PRODUCT THEOREM (Paper E, Score 8.5)
   "We prove that the SJ vacuum on CDT decomposes as a Kronecker product,
   explaining why c=1. On causets, this decomposition fails, explaining
   why c diverges."

   WHY #1: This is the only result that EXPLAINS a major open problem
   (why does CDT work and causets don't?). It gives a clean mathematical
   theorem with deep physical implications. The talk would be:
   - 5 min: c_eff divergence problem (everyone knows it)
   - 10 min: Kronecker theorem + proof sketch
   - 5 min: implications (foliation is key, proposed fixes)
   - 5 min: exact eigenvalue formula mu_k = cot(pi(2k-1)/(2T))

   Audience reaction: "That's neat and explains something we've been stuck on."

2. EXACT COMBINATORICS COMPILATION (Paper G, Score 8.0)
   "15+ exact theorems for random 2-orders including master interval formula,
   E[maximal]=H_N, antichain~2*sqrt(N), E[S_Glaser]=1."

   WHY #2: Impressive breadth, useful as null-model predictions, mathematically
   elegant. But less "wow" than explaining c_eff divergence.

3. GRAM IDENTITY UNIVERSALITY (Paper C, Score 8.0)
   "(-Delta^2)_ij = (4/N^2) * kappa_ij holds EXACTLY on ALL causal sets."

   WHY #3: Beautiful exact result linking different mathematical objects.
   But the physical interpretation (ER=EPR) is on shakier ground at large N.

4. GUE UNIVERSALITY + PHASE MIXING WARNING (Paper D, Score 8.0)
   "GUE spectral statistics are universal to ALL antisymmetric matrices,
   and the previously reported sub-Poisson behavior was a phase-mixing artifact."

   WHY #4: Important correction + null model understanding. But it's a
   negative result (GUE is generic, not special to causets).

RECOMMENDATION: Present the Kronecker product theorem (Paper E).
It's the one result that addresses a real open problem, has a clean theorem
statement, and suggests a path forward. The audience would leave with both
a "cool result" and a "research direction."

ALTERNATIVE STRATEGY: If the audience is more mathematical, present Paper G
(combinatorics). If more phenomenological, present Paper B2 (everpresent Lambda
constraints from DESI).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


# ============================================================
# SUMMARY
# ============================================================
print("=" * 80)
print("EXPERIMENT 115 SUMMARY: FIELD IMPACT (Ideas 661-670)")
print("=" * 80)
print("""
IDEA 661: Foliated Vacuum Prescription — SCORE: 7.5/10
  Foliation projection via longest-chain layering. Tests whether imposing
  approximate time slices makes c_eff finite. If yes, this is a concrete
  new vacuum state for causets.

IDEA 662: Modified BD Action with Exact Background Subtraction — SCORE: 7.0/10
  Uses master formula to subtract combinatorial noise from BD action.
  Tests whether transition becomes sharper.

IDEA 663: Fiedler-Filtered MCMC — SCORE: 7.5/10
  New MCMC prescription: accept only if Fiedler value exceeds threshold.
  Should select more manifold-like causets. Algorithmic contribution.

IDEA 664: Kurtosis Excess as New Observable — SCORE: 6.5/10
  Deviation of eigenvalue density from semicircle encodes geometry.
  N-scaling and d-dependence measured. Complementary to existing observables.

IDEA 665: Hasse Diameter Saturation — SCORE: 6.0/10
  Diameter ~ N^alpha with alpha << 0.5. "Ultrasmall-world" property.
  Cute observation but likely a corollary of link fraction scaling.

IDEA 666: Textbook Chapter — SCORE: 7.5/10
  15+ exact theorems ready for compilation. Would be definitive reference.

IDEA 667: Phase-Mixing Warning Note — SCORE: 7.0/10
  Methodological lesson for MCMC + spectral statistics community.

IDEA 668: c_eff Divergence Research Programme — SCORE: 8.5/10 (BEST)
  Diagnosis of THE central problem in SJ vacuum theory + 5 proposed fixes.
  Would get Sorkin's and Dowker's attention.

IDEA 669: Community Interest Analysis — (assessment, not scored)
  Kronecker theorem and exact combinatorics have broadest appeal.
  Different researchers value different results.

IDEA 670: The ONE Workshop Talk — RECOMMENDATION: Kronecker Product Theorem
  Explains why CDT works and causets don't. Clean theorem + physical implications.

HEADLINE: The Kronecker product theorem (Paper E) and the c_eff divergence
research programme (Idea 668) would have the most field impact. The causal set
community would take notice because these address the biggest open problem
in the field: why the SJ vacuum gives the wrong central charge on causets.
""")
sys.stdout.flush()

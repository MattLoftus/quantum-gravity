"""
Experiment 60: Round 8 — Pure Causal Set Geometry (Ideas 121-130)

Key lesson from 100 ideas: the SJ vacuum is a dead end at toy scale.
Focus on PURE CAUSAL SET GEOMETRY — properties of the partial order itself.

Ideas:
  121. Longest chain / longest antichain RATIO as dimension estimator
  122. Interval distribution MOMENTS and their scaling with N
  123. Order dimension approximation for d-dimensional sprinklings
  124. Causal set ENTROPY from BD partition function
  125. Width of the BD transition vs N (scaling exponent)
  126. Latent heat: action jump at beta_c vs N
  127. Metastability: hysteresis at BD transition (beta up vs down)
  128. Correlation between BD action and geometric observables
  129. The COMPLEMENT causal set: properties of (u, reverse(v))
  130. Finite-size scaling EXPONENTS at the BD transition
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats
from causal_sets.two_orders import TwoOrder, swap_move, bd_action_2d_fast
from causal_sets.fast_core import FastCausalSet, sprinkle_fast
from causal_sets.d_orders import DOrder, interval_entropy, bd_action_4d_fast
from causal_sets.two_orders_v2 import bd_action_corrected, mcmc_corrected
from causal_sets.bd_action import count_intervals_by_size

np.set_printoptions(precision=4, suppress=True, linewidth=120)
rng = np.random.default_rng(42)


# ============================================================
# UTILITIES
# ============================================================

def longest_antichain_size(cs):
    """
    Longest antichain = width of the poset.
    By Dilworth's theorem, width = minimum number of chains covering the poset.
    For small N, we use a greedy approach via the complement graph's max clique,
    which equals max independent set in comparability graph.

    Actually: width = N - longest_chain in the TRANSITIVE CLOSURE of the
    comparability graph? No. Use Dilworth: width = min chain decomposition.

    Simpler: width = max antichain. For a DAG, this equals the minimum
    path cover by Konig/Dilworth. Use maximum matching in bipartite graph.

    For small N, just use: build complement of comparability, find max clique.
    But max clique is NP-hard. For our small N (<= 200), brute force works
    via a greedy approach on the DAG structure.

    Actually, for a DAG (which our causet is), the longest antichain equals
    N minus the size of the maximum matching in the bipartite graph of relations.
    We can compute this via: width = N - max_matching.
    """
    N = cs.n
    if N == 0:
        return 0

    # Use the fact that for a DAG, width = N - maximum matching
    # where matching is in the bipartite representation.
    # Alternatively, for small N, enumerate antichains greedily.

    # For correctness: use the Mirsky theorem approach.
    # Partition into antichains by rank. The max antichain is at least
    # as large as the largest rank level.

    # Compute rank of each element (length of longest chain ending at it)
    order = cs.order
    rank = np.ones(N, dtype=int)
    for j in range(N):
        preds = np.where(order[:j, j])[0]
        if len(preds) > 0:
            rank[j] = np.max(rank[preds]) + 1

    # The largest rank level gives a lower bound on width
    # For a random sprinkling, this is often close to the true width
    from collections import Counter
    rank_counts = Counter(rank.tolist())
    return max(rank_counts.values())


def longest_chain(cs):
    """Longest chain in the causet."""
    return cs.longest_chain()


def ordering_fraction(cs):
    """Fraction of pairs that are related."""
    return cs.ordering_fraction()


def interval_moments(cs, max_k=20, n_moments=4):
    """Compute moments of the interval size distribution."""
    counts = count_intervals_by_size(cs, max_size=max_k)
    sizes = []
    for k in range(max_k + 1):
        sizes.extend([k] * counts.get(k, 0))
    if len(sizes) == 0:
        return [0.0] * n_moments
    sizes = np.array(sizes, dtype=float)
    moments = []
    for m in range(1, n_moments + 1):
        moments.append(np.mean(sizes ** m))
    return moments


def run_mcmc_scan(N, eps, betas, n_steps=30000, n_therm=15000, record_every=20):
    """Run MCMC at multiple beta values, return mean actions and variances."""
    results = {}
    for beta in betas:
        r = mcmc_corrected(N, beta, eps, n_steps=n_steps, n_therm=n_therm,
                           record_every=record_every, rng=np.random.default_rng(42))
        results[beta] = {
            'mean_S': np.mean(r['actions']),
            'var_S': np.var(r['actions']),
            'std_S': np.std(r['actions']),
            'actions': r['actions'],
            'samples': r['samples'],
            'accept_rate': r['accept_rate'],
        }
    return results


print("=" * 80)
print("EXPERIMENT 60: ROUND 8 — PURE CAUSAL SET GEOMETRY")
print("=" * 80)


# ============================================================
# IDEA 121: Chain/Antichain ratio as dimension estimator
# ============================================================
print("\n" + "=" * 80)
print("IDEA 121: Longest chain / longest antichain RATIO vs dimension")
print("=" * 80)
print("""
Theory: For N points sprinkled into d-dim Minkowski diamond,
  longest chain ~ c_d * N^{1/d}
  longest antichain ~ a_d * N^{(d-1)/d}
  ratio = chain/antichain ~ (c_d/a_d) * N^{1/d - (d-1)/d} = ... * N^{(2-d)/d}

For d=2: ratio ~ const (chain ~ N^{1/2}, antichain ~ N^{1/2})
For d=3: ratio ~ N^{-1/3}
For d=4: ratio ~ N^{-1/2}
For d=5: ratio ~ N^{-3/5}

The EXPONENT (2-d)/d should encode dimension.
""")

# Test across dimensions
dims = [2, 3, 4]
Ns = [30, 50, 80, 120]
n_trials = 20

print(f"{'dim':>3} {'N':>5} {'chain':>8} {'antichain':>10} {'ratio':>8} {'log_ratio':>10}")
print("-" * 55)

dim_data = {d: {'Ns': [], 'log_Ns': [], 'log_ratios': []} for d in dims}

for d in dims:
    for N in Ns:
        chains = []
        antichains = []
        ratios = []
        for trial in range(n_trials):
            cs, coords = sprinkle_fast(N, dim=d, rng=np.random.default_rng(1000*d + 100*N + trial))
            c = longest_chain(cs)
            a = longest_antichain_size(cs)
            chains.append(c)
            antichains.append(a)
            if a > 0:
                ratios.append(c / a)

        mean_c = np.mean(chains)
        mean_a = np.mean(antichains)
        mean_r = np.mean(ratios)

        dim_data[d]['Ns'].append(N)
        dim_data[d]['log_Ns'].append(np.log(N))
        dim_data[d]['log_ratios'].append(np.log(mean_r))

        print(f"{d:>3} {N:>5} {mean_c:>8.1f} {mean_a:>10.1f} {mean_r:>8.3f} {np.log(mean_r):>10.3f}")

# Fit exponent for each dimension
print("\nFitting log(ratio) = alpha * log(N) + const:")
print(f"{'dim':>3} {'alpha':>8} {'expected':>10} {'match':>6}")
print("-" * 35)
for d in dims:
    if len(dim_data[d]['log_Ns']) >= 3:
        slope, intercept, r, p, se = stats.linregress(
            dim_data[d]['log_Ns'], dim_data[d]['log_ratios'])
        expected = (2 - d) / d
        match = "YES" if abs(slope - expected) < 0.3 else "NO"
        print(f"{d:>3} {slope:>8.3f} {expected:>10.3f} {match:>6}")

# Null model: random DAG with matched density
print("\nNull model: random 2-orders (NOT sprinklings) at same N")
for N in [50, 100]:
    chains_null = []
    antichains_null = []
    for trial in range(20):
        to = TwoOrder(N, rng=np.random.default_rng(5000 + trial))
        cs = to.to_causet()
        chains_null.append(longest_chain(cs))
        antichains_null.append(longest_antichain_size(cs))
    r_null = np.mean(chains_null) / np.mean(antichains_null)
    print(f"  N={N}: chain={np.mean(chains_null):.1f}, antichain={np.mean(antichains_null):.1f}, ratio={r_null:.3f}")
    print(f"  (2-order is always d=2 embedding, so ratio should be ~const like d=2 sprinkling)")


# ============================================================
# IDEA 122: Interval distribution moments
# ============================================================
print("\n" + "=" * 80)
print("IDEA 122: Interval distribution MOMENTS and scaling with N")
print("=" * 80)
print("""
For a d-dim sprinkling of N points, interval sizes follow a distribution
whose moments should scale with N in a dimension-dependent way.
If typical interval ~ N^{2/d}, then k-th moment ~ N^{2k/d}.
Test: compute moments M1..M4 for d=2,3,4 at various N.
""")

dims_122 = [2, 3, 4]
Ns_122 = [30, 50, 80, 120]
n_trials_122 = 15

for d in dims_122:
    print(f"\n  Dimension d={d}:")
    print(f"  {'N':>5} {'M1':>8} {'M2':>10} {'M3':>12} {'M4':>14} {'M2/M1^2':>9}")
    print("  " + "-" * 65)

    moment_data = {k: [] for k in range(1, 5)}
    log_Ns = []

    for N in Ns_122:
        m_all = [[] for _ in range(4)]
        for trial in range(n_trials_122):
            cs, _ = sprinkle_fast(N, dim=d, rng=np.random.default_rng(2000*d + 100*N + trial))
            moms = interval_moments(cs, max_k=min(N, 20))
            for k in range(4):
                m_all[k].append(moms[k])

        means = [np.mean(m_all[k]) for k in range(4)]
        log_Ns.append(np.log(N))
        for k in range(4):
            moment_data[k+1].append(np.log(max(means[k], 1e-10)))

        ratio = means[1] / max(means[0]**2, 1e-10)
        print(f"  {N:>5} {means[0]:>8.2f} {means[1]:>10.1f} {means[2]:>12.0f} {means[3]:>14.0f} {ratio:>9.3f}")

    # Fit scaling exponents
    print(f"  Scaling exponents (log M_k vs log N):")
    for k in range(1, 5):
        if len(log_Ns) >= 3:
            slope, _, r, _, _ = stats.linregress(log_Ns, moment_data[k])
            expected = 2 * k / d  # naive expectation
            print(f"    M{k}: exponent = {slope:.3f} (naive expected: {expected:.3f}, R^2 = {r**2:.3f})")

# Null: random 2-orders
print("\nNull: random 2-orders (d=2 always)")
for N in [50, 100]:
    m_all = [[] for _ in range(4)]
    for trial in range(20):
        to = TwoOrder(N, rng=np.random.default_rng(7000 + trial))
        cs = to.to_causet()
        moms = interval_moments(cs, max_k=min(N, 20))
        for k in range(4):
            m_all[k].append(moms[k])
    means = [np.mean(m_all[k]) for k in range(4)]
    print(f"  N={N}: M1={means[0]:.2f}, M2={means[1]:.1f}, M2/M1^2={means[1]/max(means[0]**2,1e-10):.3f}")


# ============================================================
# IDEA 123: Order dimension
# ============================================================
print("\n" + "=" * 80)
print("IDEA 123: Order dimension of d-dimensional sprinklings")
print("=" * 80)
print("""
Order dimension = minimum number of total orders whose intersection gives the partial order.
For a 2-order, the order dimension is exactly 2 (by construction).
For a d-order, the order dimension is at most d.
For a sprinkling into d-dim Minkowski, the order dimension should be ~d.

Computing exact order dimension is NP-hard, but we can:
  1. Check that d-orders have order dim <= d (trivially true by construction)
  2. Check that sprinklings into d-dim have order dim > d-1 (they can't be represented
     as (d-1)-orders) — test by checking if the causet from a d-dim sprinkling
     matches any (d-1)-order
  3. Use ordering fraction as a proxy: for d-orders, ordering_fraction ~ 1/d!
     (since the probability that d random ranks are all concordant is 1/d!)
""")

# Test ordering fraction as dimension proxy
dims_123 = [2, 3, 4, 5]
N_123 = 80
n_trials_123 = 30

print(f"Ordering fraction for random d-orders (N={N_123}):")
print(f"  {'d':>3} {'ord_frac':>10} {'expected 1/d!':>14} {'ratio':>8}")
print("  " + "-" * 40)

for d in dims_123:
    fracs = []
    for trial in range(n_trials_123):
        do = DOrder(d, N_123, rng=np.random.default_rng(3000*d + trial))
        cs = do.to_causet_fast()
        fracs.append(ordering_fraction(cs))

    mean_frac = np.mean(fracs)
    expected = 1.0 / np.math.factorial(d)
    ratio = mean_frac / expected
    print(f"  {d:>3} {mean_frac:>10.4f} {expected:>14.4f} {ratio:>8.3f}")

# Compare with sprinklings
print(f"\nOrdering fraction for Minkowski sprinklings (N={N_123}):")
for d in [2, 3, 4]:
    fracs = []
    for trial in range(n_trials_123):
        cs, _ = sprinkle_fast(N_123, dim=d, rng=np.random.default_rng(4000*d + trial))
        fracs.append(ordering_fraction(cs))
    mean_frac = np.mean(fracs)
    print(f"  d={d}: ord_frac = {mean_frac:.4f}")

print("\nKey insight: ordering fraction 1/d! is KNOWN (Brightwell-Gregory).")
print("The question is whether d-orders faithfully reproduce sprinkling statistics.")


# ============================================================
# IDEA 124: Causal set entropy from BD partition function
# ============================================================
print("\n" + "=" * 80)
print("IDEA 124: Causal set entropy from BD partition function")
print("=" * 80)
print("""
Z(beta) = sum_C exp(-beta * S_BD[C])
ln Z = -beta * <S> + S_entropy  (where S_entropy is the Boltzmann entropy)

From MCMC: we can estimate <S>(beta), var(S)(beta), and from that
  specific heat C_v = beta^2 * var(S)
  entropy via thermodynamic integration: S(beta) = S(0) + integral_0^beta C_v/beta' dbeta'

At beta=0: S(0) = ln(number of 2-orders) = ln(N!^2) ~ 2N ln N (for 2-orders)

Test: compute entropy vs N and check for area law or volume law.
For 2D causal sets, "area" = boundary size ~ N^{1/2}, "volume" = N.
""")

eps = 0.12
Ns_124 = [20, 30, 40]

print(f"Specific heat C_v = beta^2 * var(S) at various beta, eps={eps}:")

for N in Ns_124:
    beta_c_est = 1.66 / (N * eps**2)
    betas = [0.0, 0.3*beta_c_est, 0.6*beta_c_est, 0.9*beta_c_est, beta_c_est, 1.2*beta_c_est]
    betas = [max(b, 0.0) for b in betas]

    print(f"\n  N={N} (beta_c ~ {beta_c_est:.2f}):")
    print(f"  {'beta':>8} {'<S>':>8} {'var(S)':>10} {'C_v':>10} {'accept':>8}")
    print("  " + "-" * 50)

    for beta in betas:
        r = mcmc_corrected(N, beta, eps, n_steps=12000, n_therm=6000,
                           record_every=20, rng=np.random.default_rng(42))
        mean_S = np.mean(r['actions'])
        var_S = np.var(r['actions'])
        Cv = beta**2 * var_S
        print(f"  {beta:>8.2f} {mean_S:>8.2f} {var_S:>10.3f} {Cv:>10.3f} {r['accept_rate']:>8.3f}")

# Entropy estimate via S(0) - beta*<S>_0 + integral
# S(0) = ln(N!^2) for 2-orders
print(f"\n  Entropy at beta=0: S(0) = ln(N!^2) = 2*ln(N!)")
for N in Ns_124:
    from math import lgamma
    S0 = 2 * lgamma(N + 1)
    print(f"    N={N}: S(0) = {S0:.1f}, S(0)/N = {S0/N:.2f}, S(0)/N^(1/2) = {S0/N**0.5:.2f}")

print("\n  If entropy ~ N (volume law), ratio S/N ~ const")
print("  If entropy ~ N^{1/2} (area law in 2D), ratio S/sqrt(N) ~ const")
print("  Result: S(0)/N ~ 2*ln(N) which grows, so entropy is super-volume at beta=0")
print("  The interesting question is entropy NEAR the transition.")


# ============================================================
# IDEA 125: Width of the BD transition vs N
# ============================================================
print("\n" + "=" * 80)
print("IDEA 125: BD transition WIDTH vs N")
print("=" * 80)
print("""
For a first-order transition, the width Delta_beta ~ 1/N.
For a second-order transition, Delta_beta ~ N^{-1/nu} where nu is the
correlation length exponent.

We measure the width as the beta range over which the specific heat
C_v = beta^2 * var(S) exceeds half its peak value.
""")

eps = 0.12
Ns_125 = [20, 30, 50]

print(f"Transition width measurement (eps={eps}):")
print(f"{'N':>5} {'beta_c':>8} {'peak_Cv':>10} {'width':>10} {'N*width':>10}")
print("-" * 50)

width_data = {'log_N': [], 'log_width': []}

for N in Ns_125:
    beta_c_est = 1.66 / (N * eps**2)
    # Fine scan around beta_c
    n_betas = 10
    betas = np.linspace(0.5 * beta_c_est, 1.8 * beta_c_est, n_betas)

    Cvs = []
    for beta in betas:
        r = mcmc_corrected(N, beta, eps, n_steps=10000, n_therm=5000,
                           record_every=15, rng=np.random.default_rng(42))
        var_S = np.var(r['actions'])
        Cvs.append(beta**2 * var_S)

    Cvs = np.array(Cvs)
    peak_idx = np.argmax(Cvs)
    peak_Cv = Cvs[peak_idx]

    # Half-max width
    half_max = peak_Cv / 2
    above = betas[Cvs >= half_max]
    if len(above) >= 2:
        width = above[-1] - above[0]
    else:
        width = betas[1] - betas[0]  # minimum resolution

    print(f"{N:>5} {betas[peak_idx]:>8.2f} {peak_Cv:>10.2f} {width:>10.4f} {N*width:>10.2f}")

    width_data['log_N'].append(np.log(N))
    width_data['log_width'].append(np.log(width))

# Fit scaling
if len(width_data['log_N']) >= 3:
    slope, intercept, r, p, se = stats.linregress(
        width_data['log_N'], width_data['log_width'])
    print(f"\nScaling: Delta_beta ~ N^{slope:.2f} (R^2 = {r**2:.3f})")
    print(f"First-order prediction: exponent = -1.0")
    print(f"Match: {'YES' if abs(slope + 1) < 0.3 else 'PARTIAL' if abs(slope + 1) < 0.6 else 'NO'}")


# ============================================================
# IDEA 126: Latent heat (action jump at beta_c)
# ============================================================
print("\n" + "=" * 80)
print("IDEA 126: Latent heat — action jump at beta_c vs N")
print("=" * 80)
print("""
For a first-order transition, the action jumps discontinuously at beta_c.
The latent heat L = Delta<S> should scale linearly with N.
Measure by comparing <S> just below and just above beta_c.
""")

eps = 0.12
Ns_126 = [20, 30, 50]

print(f"Latent heat measurement (eps={eps}):")
print(f"{'N':>5} {'beta_c':>8} {'<S>_low':>10} {'<S>_high':>10} {'Delta_S':>10} {'Delta_S/N':>10}")
print("-" * 60)

latent_data = {'N': [], 'delta_S': []}

for N in Ns_126:
    beta_c_est = 1.66 / (N * eps**2)

    # Below transition
    r_low = mcmc_corrected(N, 0.7 * beta_c_est, eps, n_steps=12000, n_therm=6000,
                            record_every=20, rng=np.random.default_rng(42))
    # Above transition
    r_high = mcmc_corrected(N, 1.5 * beta_c_est, eps, n_steps=12000, n_therm=6000,
                             record_every=20, rng=np.random.default_rng(42))

    S_low = np.mean(r_low['actions'])
    S_high = np.mean(r_high['actions'])
    delta_S = abs(S_high - S_low)

    print(f"{N:>5} {beta_c_est:>8.2f} {S_low:>10.2f} {S_high:>10.2f} {delta_S:>10.2f} {delta_S/N:>10.4f}")
    latent_data['N'].append(N)
    latent_data['delta_S'].append(delta_S)

# Fit scaling
if len(latent_data['N']) >= 3:
    slope, intercept, r, p, se = stats.linregress(
        np.log(latent_data['N']), np.log(np.array(latent_data['delta_S']) + 1e-10))
    print(f"\nScaling: Delta_S ~ N^{slope:.2f} (R^2 = {r**2:.3f})")
    print(f"First-order prediction: exponent = 1.0")


# ============================================================
# IDEA 127: Metastability / Hysteresis
# ============================================================
print("\n" + "=" * 80)
print("IDEA 127: Metastability — hysteresis at BD transition")
print("=" * 80)
print("""
At a first-order transition, scanning beta UP (from disordered to ordered)
vs DOWN (from ordered to disordered) should show hysteresis: the transition
happens at different apparent beta_c values.
""")

eps = 0.12
N_127 = 30
beta_c_est = 1.66 / (N_127 * eps**2)
n_betas = 10
betas_scan = np.linspace(0.3 * beta_c_est, 2.0 * beta_c_est, n_betas)

print(f"N={N_127}, eps={eps}, beta_c ~ {beta_c_est:.2f}")
print(f"Scanning {n_betas} beta values from {betas_scan[0]:.2f} to {betas_scan[-1]:.2f}")

# Scan UP: start from random (disordered) state
print("\n  Scan UP (starting disordered):")
current_up = TwoOrder(N_127, rng=np.random.default_rng(42))
S_up = []

for beta in betas_scan:
    current_cs = current_up.to_causet()
    current_S = bd_action_corrected(current_cs, eps)
    actions_local = []

    for step in range(6000):
        proposed = swap_move(current_up, rng)
        proposed_cs = proposed.to_causet()
        proposed_S = bd_action_corrected(proposed_cs, eps)
        dS = beta * (proposed_S - current_S)
        if dS <= 0 or rng.random() < np.exp(-min(dS, 500)):
            current_up = proposed
            current_cs = proposed_cs
            current_S = proposed_S
        if step >= 3000 and step % 20 == 0:
            actions_local.append(current_S)

    S_up.append(np.mean(actions_local))

# Scan DOWN: start from ordered (crystalline) state — identity permutations
print("  Scan DOWN (starting ordered):")
current_down = TwoOrder.from_permutations(np.arange(N_127), np.arange(N_127))
S_down = []

for beta in reversed(betas_scan):
    current_cs = current_down.to_causet()
    current_S = bd_action_corrected(current_cs, eps)
    actions_local = []

    for step in range(6000):
        proposed = swap_move(current_down, rng)
        proposed_cs = proposed.to_causet()
        proposed_S = bd_action_corrected(proposed_cs, eps)
        dS = beta * (proposed_S - current_S)
        if dS <= 0 or rng.random() < np.exp(-min(dS, 500)):
            current_down = proposed
            current_cs = proposed_cs
            current_S = proposed_S
        if step >= 3000 and step % 20 == 0:
            actions_local.append(current_S)

    S_down.append(np.mean(actions_local))

S_down = list(reversed(S_down))

print(f"\n  {'beta':>8} {'<S>_up':>10} {'<S>_down':>10} {'gap':>10}")
print("  " + "-" * 42)
for i, beta in enumerate(betas_scan):
    gap = abs(S_up[i] - S_down[i])
    marker = " <-- HYSTERESIS" if gap > 0.5 else ""
    print(f"  {beta:>8.2f} {S_up[i]:>10.2f} {S_down[i]:>10.2f} {gap:>10.3f}{marker}")

max_gap = max(abs(S_up[i] - S_down[i]) for i in range(len(betas_scan)))
print(f"\n  Maximum hysteresis gap: {max_gap:.3f}")
print(f"  Significant hysteresis (>1.0): {'YES' if max_gap > 1.0 else 'NO'}")


# ============================================================
# IDEA 128: Correlation between action and geometric observables
# ============================================================
print("\n" + "=" * 80)
print("IDEA 128: Correlation between BD action and geometric observables")
print("=" * 80)
print("""
Which geometric quantity best predicts the BD action?
Candidates: ordering fraction, longest chain, interval entropy, number of links.
Test at beta=0 (uniform sampling of 2-orders).
""")

N_128 = 40
eps_128 = 0.12
n_samples = 200

actions_128 = []
ord_fracs = []
chains_128 = []
entropies_128 = []
link_counts = []

print(f"Generating {n_samples} random 2-orders at N={N_128}...")
for i in range(n_samples):
    to = TwoOrder(N_128, rng=np.random.default_rng(8000 + i))
    cs = to.to_causet()

    S = bd_action_corrected(cs, eps_128)
    actions_128.append(S)
    ord_fracs.append(ordering_fraction(cs))
    chains_128.append(longest_chain(cs))
    entropies_128.append(interval_entropy(cs))

    links = cs.link_matrix()
    link_counts.append(int(np.sum(links)))

actions_128 = np.array(actions_128)
ord_fracs = np.array(ord_fracs)
chains_128 = np.array(chains_128)
entropies_128 = np.array(entropies_128)
link_counts = np.array(link_counts)

print(f"\nCorrelations with BD action (Pearson r):")
observables = {
    'ordering_fraction': ord_fracs,
    'longest_chain': chains_128,
    'interval_entropy': entropies_128,
    'link_count': link_counts,
}

for name, obs in observables.items():
    r_val, p_val = stats.pearsonr(actions_128, obs)
    rho, _ = stats.spearmanr(actions_128, obs)
    print(f"  {name:>25}: Pearson r = {r_val:+.4f} (p={p_val:.2e}), Spearman rho = {rho:+.4f}")

print(f"\n  Best predictor of BD action: {max(observables.keys(), key=lambda k: abs(stats.pearsonr(actions_128, observables[k])[0]))}")


# ============================================================
# IDEA 129: The COMPLEMENT causal set
# ============================================================
print("\n" + "=" * 80)
print("IDEA 129: The COMPLEMENT causal set — (u, reverse(v))")
print("=" * 80)
print("""
Given a 2-order with permutations (u, v), define the complement as (u, reverse(v))
where reverse(v) = N-1-v. The complement has "orthogonal" causal structure.

If (u, v) gives a causet C, then (u, N-1-v) gives a causet C' where:
  i < j in C  iff  u[i] < u[j] AND v[i] < v[j]
  i < j in C' iff  u[i] < u[j] AND v[i] > v[j]

So C and C' have complementary causal structure in the v-direction.
The UNION of relations in C and C' gives all pairs where u[i] < u[j],
which is a total order. So |relations(C)| + |relations(C')| = N(N-1)/2.

Question: How do geometric properties of C compare with C'?
""")

N_129 = 60
n_trials_129 = 50

print(f"Comparing 2-order with its complement (N={N_129}, {n_trials_129} trials):")
print(f"  {'property':>25} {'original':>10} {'complement':>12} {'corr':>8}")
print("  " + "-" * 60)

orig_ords = []
comp_ords = []
orig_chains = []
comp_chains = []
orig_actions = []
comp_actions = []
orig_entropies = []
comp_entropies = []

for trial in range(n_trials_129):
    to = TwoOrder(N_129, rng=np.random.default_rng(9000 + trial))
    cs_orig = to.to_causet()

    # Complement: reverse the v permutation
    v_complement = N_129 - 1 - to.v
    to_comp = TwoOrder.from_permutations(to.u, v_complement)
    cs_comp = to_comp.to_causet()

    orig_ords.append(ordering_fraction(cs_orig))
    comp_ords.append(ordering_fraction(cs_comp))
    orig_chains.append(longest_chain(cs_orig))
    comp_chains.append(longest_chain(cs_comp))
    orig_actions.append(bd_action_corrected(cs_orig, 0.12))
    comp_actions.append(bd_action_corrected(cs_comp, 0.12))
    orig_entropies.append(interval_entropy(cs_orig))
    comp_entropies.append(interval_entropy(cs_comp))

# Check the complementarity relation
total_rels = np.array(orig_ords) + np.array(comp_ords)
print(f"  {'ord_frac (orig+comp)':>25} {np.mean(total_rels):>10.4f} {'(should be ~1.0)':>12}")

properties = {
    'ordering_fraction': (orig_ords, comp_ords),
    'longest_chain': (orig_chains, comp_chains),
    'BD_action': (orig_actions, comp_actions),
    'interval_entropy': (orig_entropies, comp_entropies),
}

for name, (orig, comp) in properties.items():
    r_val, p_val = stats.pearsonr(orig, comp)
    print(f"  {name:>25} {np.mean(orig):>10.3f} {np.mean(comp):>12.3f} {r_val:>8.4f}")

print(f"\n  Key finding: ordering fractions sum to ~{np.mean(total_rels):.4f}")
print(f"  (Expected 1.0 since C and C' partition all concordant pairs of the u-order)")


# ============================================================
# IDEA 130: Finite-size scaling exponents at BD transition
# ============================================================
print("\n" + "=" * 80)
print("IDEA 130: Finite-size scaling EXPONENTS at BD transition")
print("=" * 80)
print("""
At a phase transition, FSS predicts:
  C_v_max ~ N^{alpha/nu*d}  (specific heat peak)
  chi_max ~ N^{gamma/nu*d}  (susceptibility peak)

For first-order transitions in the 2-order model:
  C_v_max ~ N^2 (since var(S) ~ N^2 and C_v = beta^2 * var(S))

Measure the scaling of the specific heat peak with N.
""")

eps = 0.12
Ns_130 = [20, 30, 40, 50]

print(f"Specific heat peak scaling (eps={eps}):")
print(f"{'N':>5} {'beta_peak':>10} {'Cv_peak':>10} {'var_S_peak':>12}")
print("-" * 42)

Cv_peaks = []
var_peaks = []
log_Ns_130 = []

for N in Ns_130:
    beta_c_est = 1.66 / (N * eps**2)
    betas = np.linspace(0.5 * beta_c_est, 2.0 * beta_c_est, 8)

    best_Cv = 0
    best_var = 0
    best_beta = 0

    for beta in betas:
        r = mcmc_corrected(N, beta, eps, n_steps=10000, n_therm=5000,
                           record_every=15, rng=np.random.default_rng(42))
        var_S = np.var(r['actions'])
        Cv = beta**2 * var_S
        if Cv > best_Cv:
            best_Cv = Cv
            best_var = var_S
            best_beta = beta

    print(f"{N:>5} {best_beta:>10.2f} {best_Cv:>10.2f} {best_var:>12.3f}")
    Cv_peaks.append(best_Cv)
    var_peaks.append(best_var)
    log_Ns_130.append(np.log(N))

# Fit scaling exponents
if len(log_Ns_130) >= 3:
    slope_Cv, _, r_Cv, _, _ = stats.linregress(log_Ns_130, np.log(np.array(Cv_peaks) + 1e-10))
    slope_var, _, r_var, _, _ = stats.linregress(log_Ns_130, np.log(np.array(var_peaks) + 1e-10))

    print(f"\nScaling exponents:")
    print(f"  C_v_max ~ N^{slope_Cv:.2f} (R^2 = {r_Cv**2:.3f})")
    print(f"  var(S)_max ~ N^{slope_var:.2f} (R^2 = {r_var**2:.3f})")
    print(f"  First-order prediction: C_v ~ N^2, var(S) ~ N^2")
    print(f"  Second-order prediction: C_v ~ N^{{alpha/nu*d}}, typically sublinear")

# Also check beta_c scaling
print(f"\n  beta_c location vs N:")
print(f"  beta_c(N) = 1.66 / (N * eps^2) = {1.66/eps**2:.1f} / N")
print(f"  This 1/N scaling is characteristic of a first-order transition.")


# ============================================================
# SUMMARY AND SCORING
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY AND SCORING")
print("=" * 80)

ideas = [
    (121, "Chain/antichain ratio as dimension estimator",
     "Tests whether ratio scales as N^{(2-d)/d}"),
    (122, "Interval distribution moments",
     "Moment scaling exponents encode dimension"),
    (123, "Order dimension / ordering fraction",
     "Confirms Brightwell-Gregory 1/d! formula"),
    (124, "Causal set entropy (BD partition function)",
     "Thermodynamic integration for entropy scaling"),
    (125, "BD transition width vs N",
     "Tests first-order prediction width ~ 1/N"),
    (126, "Latent heat: action jump at beta_c",
     "Tests first-order prediction DeltaS ~ N"),
    (127, "Metastability / hysteresis",
     "Scans beta up vs down for hysteresis"),
    (128, "Action-geometry correlation",
     "Which observable best predicts BD action?"),
    (129, "Complement causal set",
     "Properties of (u, reverse(v)) vs (u, v)"),
    (130, "Finite-size scaling exponents",
     "Cv peak scaling at BD transition"),
]

print(f"\n{'ID':>4} {'Idea':>45} {'Score':>6} {'Assessment'}")
print("-" * 100)

for id_num, name, desc in ideas:
    print(f"\n  Idea {id_num}: {name}")
    print(f"  Description: {desc}")

print("\n\nDONE. See detailed results above for each idea.")

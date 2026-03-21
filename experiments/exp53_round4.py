"""
Experiment 53: Ideas 76-85 — Round 4 of 8+ Search

Ten genuinely new ideas, focused on what's UNIQUE to causets:
- Lorentzian signature (causal order vs spatial distance)
- Sprinkled causets vs 2-orders (embedding information)
- Phase transition critical phenomena
- Causal matrix C as a direct object (not just via W)
- Higher-point / connected correlators
- Dynamics of W under MCMC evolution

Methodology: quick test, null model control, honest 1-10 score.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats
from causal_sets.two_orders import TwoOrder, swap_move
from causal_sets.fast_core import FastCausalSet, sprinkle_fast
import time

rng = np.random.default_rng(42)


def sj_full(cs):
    """Return W and eigendecomposition of i*Delta."""
    N = cs.n
    C = cs.order.astype(float)
    Delta = (2.0 / N) * (C.T - C)
    H = 1j * Delta
    evals, evecs = np.linalg.eigh(H)
    evals = evals.real
    pos = evals > 1e-12
    if not np.any(pos):
        return np.zeros((N, N)), evals, evecs
    W = (evecs[:, pos] @ np.diag(evals[pos]) @ evecs[:, pos].conj().T).real
    w_max = np.linalg.eigvalsh(W).max()
    if w_max > 1:
        W = W / w_max
    return W, evals, evecs


def entanglement_entropy(W, region):
    if len(region) < 2:
        return 0.0
    W_A = W[np.ix_(region, region)]
    eigs = np.linalg.eigvalsh(W_A)
    eigs = np.clip(eigs, 1e-15, 1 - 1e-15)
    return float(-np.sum(eigs * np.log(eigs) + (1 - eigs) * np.log(1 - eigs)))


def random_dag(N, density, rng_local):
    """Random DAG with given relation density (null model)."""
    cs = FastCausalSet(N)
    for i in range(N):
        for j in range(i + 1, N):
            if rng_local.random() < density:
                cs.order[i, j] = True
    # Transitive closure
    for k in range(N):
        for i in range(N):
            if cs.order[i, k]:
                for j in range(k + 1, N):
                    if cs.order[k, j]:
                        cs.order[i, j] = True
    return cs


# ================================================================
print("=" * 75)
print("IDEA 76: SJ VACUUM ON SPRINKLED CAUSETS vs 2-ORDERS")
print("Does knowing the EMBEDDING change the SJ vacuum structure?")
print("Sprinkled causets come with coordinates; 2-orders don't.")
print("Compare: is S(half) / ln(N) different for sprinkled vs random 2-orders?")
print("=" * 75)

# The key insight: sprinkled causets are 2-orders that faithfully embed in
# Minkowski space. Random 2-orders are generic. If the SJ vacuum "knows"
# about the embedding (via the causal structure), sprinkled causets should
# have c_eff closer to 1 (the CFT value).

Ns_76 = [40, 60, 80, 100]
n_trials = 5

print("\n  Sprinkled causets (Minkowski diamond):")
for N in Ns_76:
    c_effs = []
    for trial in range(n_trials):
        cs, coords = sprinkle_fast(N, dim=2, extent_t=1.0, rng=rng)
        W, _, _ = sj_full(cs)
        S_half = entanglement_entropy(W, list(range(N // 2)))
        c_eff = 3 * S_half / np.log(N)
        c_effs.append(c_eff)
    print(f"  N={N:3d}: c_eff = {np.mean(c_effs):.3f} +/- {np.std(c_effs):.3f}")

print("\n  Random 2-orders:")
for N in Ns_76:
    c_effs = []
    for trial in range(n_trials):
        to = TwoOrder(N, rng=rng)
        cs = to.to_causet()
        W, _, _ = sj_full(cs)
        S_half = entanglement_entropy(W, list(range(N // 2)))
        c_eff = 3 * S_half / np.log(N)
        c_effs.append(c_eff)
    print(f"  N={N:3d}: c_eff = {np.mean(c_effs):.3f} +/- {np.std(c_effs):.3f}")

# Also: check ordering fraction (tells us density)
print("\n  Ordering fractions:")
for N in [60]:
    ofs_sprinkle = []
    ofs_random = []
    for trial in range(n_trials):
        cs_s, _ = sprinkle_fast(N, dim=2, extent_t=1.0, rng=rng)
        ofs_sprinkle.append(cs_s.ordering_fraction())
        to = TwoOrder(N, rng=rng)
        cs_r = to.to_causet()
        ofs_random.append(cs_r.ordering_fraction())
    print(f"  Sprinkled (N={N}): of = {np.mean(ofs_sprinkle):.3f} +/- {np.std(ofs_sprinkle):.3f}")
    print(f"  Random 2-order (N={N}): of = {np.mean(ofs_random):.3f} +/- {np.std(ofs_random):.3f}")

print("\n  SCORE 76: If c_eff differs significantly between sprinkled and random,")
print("  that shows the SJ vacuum is sensitive to geometric embedding, not just")
print("  causal structure. This would be a genuinely new finding.")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 77: CAUSAL DIAMOND COUNTING — 'VOLUME OF PAST' DISTRIBUTION")
print("The distribution P(|past(x)|) for element x encodes geometry.")
print("For d-dim Minkowski: P(|past|) has a specific shape.")
print("Is this shape preserved by the SJ vacuum? I.e., does W 'know' volumes?")
print("=" * 75)

# For each element x, compute |past(x)| = number of elements y with y < x.
# Also compute the "SJ weight" of x: sum_y W(x, y) for y in past(x).
# If W encodes geometry, the SJ weight should correlate with |past(x)|.

N = 80
to = TwoOrder(N, rng=rng)
cs = to.to_causet()
W, evals, _ = sj_full(cs)

past_sizes = []
sj_weights = []
for x in range(N):
    past = [y for y in range(N) if cs.order[y, x]]
    past_size = len(past)
    if past_size > 0:
        w_past = sum(abs(W[x, y]) for y in past)
    else:
        w_past = 0.0
    past_sizes.append(past_size)
    sj_weights.append(w_past)

r_vol = np.corrcoef(past_sizes, sj_weights)[0, 1]
print(f"\n  Correlation between |past(x)| and SJ-weight of past: r = {r_vol:.3f}")

# Null: random symmetric matrix W
W_null = rng.standard_normal((N, N))
W_null = (W_null + W_null.T) / 2
W_null = np.abs(W_null) * 0.01
sj_null = []
for x in range(N):
    past = [y for y in range(N) if cs.order[y, x]]
    if len(past) > 0:
        sj_null.append(sum(abs(W_null[x, y]) for y in past))
    else:
        sj_null.append(0.0)

r_null = np.corrcoef(past_sizes, sj_null)[0, 1]
print(f"  Null (random W): r = {r_null:.3f}")
print(f"  Gap: {r_vol - r_null:.3f}")
print(f"\n  If gap >> 0: W encodes volume information beyond just 'more elements = more sum'")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 78: CAUSAL MATRIX SPECTRUM vs SJ SPECTRUM")
print("The causal matrix C itself is a random 0/1 matrix with rich structure.")
print("Compare spectrum of C with spectrum of the SJ Hermitian H = i*(2/N)(C^T-C).")
print("Is the SJ spectrum a 'filtered' version of C's spectrum?")
print("=" * 75)

# Nobody has compared the spectral properties of C directly with those of H.
# C is a triangular 0/1 matrix. H is derived from C. But the relationship
# between their spectra is non-trivial.

N = 100
to = TwoOrder(N, rng=rng)
cs = to.to_causet()
C = cs.order.astype(float)

# Spectrum of C (as a real matrix)
evals_C = np.linalg.eigvals(C)
# C is upper triangular (after topological sort) → eigenvalues are all 0
# Actually not exactly — it's not in natural labeling necessarily
print(f"\n  Spectrum of C:")
print(f"    |evals| range: [{np.min(np.abs(evals_C)):.4f}, {np.max(np.abs(evals_C)):.4f}]")
print(f"    Real parts: [{np.min(evals_C.real):.4f}, {np.max(evals_C.real):.4f}]")

# Better: spectrum of C + C^T (symmetrized adjacency of causal relations)
A_causal = C + C.T  # symmetric
evals_A = np.linalg.eigvalsh(A_causal)
print(f"  Spectrum of C + C^T (symmetrized):")
print(f"    Range: [{evals_A.min():.2f}, {evals_A.max():.2f}]")
print(f"    Mean: {evals_A.mean():.2f}, Std: {evals_A.std():.2f}")

# SJ spectrum
H = 1j * (2.0 / N) * (C.T - C)
evals_H = np.linalg.eigvalsh(H).real
pos_H = evals_H[evals_H > 1e-12]
print(f"  SJ positive spectrum:")
print(f"    Count: {len(pos_H)}/{N}")
print(f"    Range: [{pos_H.min():.4f}, {pos_H.max():.4f}]")

# Key test: is there a spectral gap in H? (between zero modes and positive)
sorted_H = np.sort(np.abs(evals_H))
gaps = np.diff(sorted_H)
max_gap_idx = np.argmax(gaps)
print(f"  Largest spectral gap at index {max_gap_idx}: gap = {gaps[max_gap_idx]:.4f}")
print(f"    Below gap: {max_gap_idx + 1} eigenvalues, above: {N - max_gap_idx - 1}")

# Compare with random DAG
cs_dag = random_dag(N, cs.ordering_fraction(), rng)
C_dag = cs_dag.order.astype(float)
H_dag = 1j * (2.0 / N) * (C_dag.T - C_dag)
evals_dag = np.linalg.eigvalsh(H_dag).real
pos_dag = evals_dag[evals_dag > 1e-12]
print(f"\n  Random DAG (matched density):")
print(f"    Positive eigenvalues: {len(pos_dag)}/{N}")
print(f"    Range: [{pos_dag.min():.4f} if len>0 else NA, {pos_dag.max():.4f} if len>0 else NA]")

# Ratio of positive eigenvalues: is this a causet-specific number?
ratio_causet = len(pos_H) / N
ratio_dag = len(pos_dag) / N
print(f"\n  Fraction of positive eigenvalues:")
print(f"    Causet: {ratio_causet:.3f}")
print(f"    Random DAG: {ratio_dag:.3f}")
print(f"  If different: the positive-frequency content is geometry-dependent")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 79: CONNECTED 4-POINT CORRELATOR OF SJ VACUUM")
print("Higher-point functions probe non-Gaussianity of the SJ state.")
print("For a truly Gaussian state, the connected 4-point function = 0.")
print("=" * 75)

# The SJ vacuum is defined as a Gaussian state. But on a discrete causal set,
# the "Gaussian" structure might break down. Test: compute the connected
# 4-point function G_c(i,j,k,l) = <phi_i phi_j phi_k phi_l> - disconnected
# For a Gaussian state: G_c = 0 exactly.
# Non-zero G_c would signal non-Gaussianity (interactions from discreteness).

N = 50
to = TwoOrder(N, rng=rng)
cs = to.to_causet()
W, _, _ = sj_full(cs)

# For a Gaussian state with 2-point function W:
# <phi_i phi_j phi_k phi_l> = W[i,j]*W[k,l] + W[i,k]*W[j,l] + W[i,l]*W[j,k]
# Connected = full - disconnected = 0 by construction.
# BUT: What if the state ISN'T truly Gaussian due to the discrete eigenvalue structure?
# Check: compute Tr(W^2) and Tr(W) and see if purity = Tr(rho^2) is consistent
# with a Gaussian state.

# For a Gaussian state of N modes with correlation matrix W:
# purity = prod_k 1/(2*nu_k*(1-nu_k)) if eigenvalues nu_k are in (0,1)
# The state IS Gaussian by construction, so the connected 4-point = 0 exactly.

# NEW ANGLE: What about the 4-point function of the SJ field evaluated at
# different CAUSAL DISTANCES? Even though G_c = 0, the structure of the
# 2-point function in the 4-point decomposition might show interesting
# clustering when evaluated over elements at specific causal distances.

# Define causal distance = longest chain length between two elements
def chain_length(cs, i, j):
    """Longest chain from i to j."""
    N = cs.n
    if not cs.order[i, j]:
        return 0
    # Dynamic programming on elements between i and j
    between = [k for k in range(N) if cs.order[i, k] and cs.order[k, j]]
    if len(between) == 0:
        return 1  # link
    # Longest chain through between
    dp = {k: 1 for k in between}
    for k in between:
        for m in between:
            if cs.order[k, m]:
                dp[m] = max(dp[m], dp[k] + 1)
    return max(dp.values()) + 1

# Compute W(i,j) as function of causal distance
dist_w = {}  # distance → list of |W[i,j]|
sample_pairs = rng.choice(N, size=(min(200, N*(N-1)//2), 2), replace=True)
for idx in range(len(sample_pairs)):
    i, j = int(sample_pairs[idx, 0]), int(sample_pairs[idx, 1])
    if i == j:
        continue
    if i > j:
        i, j = j, i
    d = chain_length(cs, i, j)
    if d > 0:
        dist_w.setdefault(d, []).append(abs(W[i, j]))

print(f"\n  |W(i,j)| vs causal distance (chain length):")
for d in sorted(dist_w.keys()):
    vals = dist_w[d]
    if len(vals) >= 3:
        print(f"    d={d}: <|W|> = {np.mean(vals):.4f} +/- {np.std(vals):.4f} (n={len(vals)})")

# Now check: does |W|^2 at distance d fall off as d^{-2*alpha}?
# This gives the field dimension Delta from <phi(x)phi(y)> ~ |x-y|^{-2*Delta}
distances = []
mean_w = []
for d in sorted(dist_w.keys()):
    if len(dist_w[d]) >= 3 and d > 0:
        distances.append(d)
        mean_w.append(np.mean(dist_w[d]))

if len(distances) >= 3:
    log_d = np.log(distances)
    log_w = np.log(mean_w)
    slope, intercept, r, p, se = stats.linregress(log_d, log_w)
    print(f"\n  Power law fit: |W| ~ d^{slope:.2f}  (r={r:.3f}, p={p:.4f})")
    print(f"  Field dimension Delta = {-slope/2:.2f}")
    print(f"  (For 2D CFT massless scalar: Delta = 0, W ~ ln(d), not power law)")
else:
    print(f"\n  Not enough distance bins for power law fit")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 80: TIMELIKE vs SPACELIKE ENTANGLEMENT ASYMMETRY")
print("Causets have a LORENTZIAN signature. Does entanglement respect it?")
print("Split the causet into 'timelike strips' vs 'spacelike strips'")
print("and compare entanglement entropy.")
print("=" * 75)

# In Minkowski space, there's a fundamental asymmetry: timelike vs spacelike.
# The SJ vacuum should encode this. For a sprinkled causet where we KNOW
# the coordinates, we can partition into:
#   - "temporal" half: elements with t > median(t) vs t < median(t)
#   - "spatial" half: elements with x > median(x) vs x < median(x)
# The entanglement across a temporal boundary should differ from that across
# a spatial boundary. This is a UNIQUELY Lorentzian signature.

N = 80
n_trials_80 = 8
S_temporal = []
S_spatial = []

for trial in range(n_trials_80):
    cs, coords = sprinkle_fast(N, dim=2, extent_t=1.0, rng=rng)
    W, _, _ = sj_full(cs)

    t = coords[:, 0]
    # For spatial coordinate in diamond, use the second coordinate
    # But sprinkle_fast only gives t coord. Reconstruct from 2-order structure.
    # Actually sprinkle_fast returns coords!
    if coords.shape[1] >= 2:
        x = coords[:, 1]
    else:
        x = np.zeros(N)

    # Temporal partition: t > median
    t_median = np.median(t)
    region_t = list(np.where(t > t_median)[0])

    # Spatial partition: x > median (works for 2D diamond)
    x_median = np.median(x)
    region_x = list(np.where(x > x_median)[0])

    S_t = entanglement_entropy(W, region_t)
    S_x = entanglement_entropy(W, region_x)
    S_temporal.append(S_t)
    S_spatial.append(S_x)

print(f"\n  Temporal partition: S = {np.mean(S_temporal):.3f} +/- {np.std(S_temporal):.3f}")
print(f"  Spatial partition:  S = {np.mean(S_spatial):.3f} +/- {np.std(S_spatial):.3f}")
ratio_ts = np.mean(S_temporal) / max(np.mean(S_spatial), 1e-15)
print(f"  Ratio S_temporal / S_spatial = {ratio_ts:.3f}")

# Null model: do the same on a random DAG (which has no Lorentzian structure)
S_t_null = []
S_x_null = []
for trial in range(n_trials_80):
    cs, coords = sprinkle_fast(N, dim=2, extent_t=1.0, rng=rng)
    # Scramble the causal order (preserving the DAG structure but breaking geometry)
    perm = rng.permutation(N)
    cs_scrambled = FastCausalSet(N)
    for i in range(N):
        for j in range(N):
            cs_scrambled.order[i, j] = cs.order[perm[i], perm[j]]
    W_s, _, _ = sj_full(cs_scrambled)
    # Use ORIGINAL coordinates for partitioning (DAG doesn't match coords)
    t = coords[:, 0]
    x = coords[:, 1] if coords.shape[1] >= 2 else np.zeros(N)
    region_t = list(np.where(t > np.median(t))[0])
    region_x = list(np.where(x > np.median(x))[0])
    S_t_null.append(entanglement_entropy(W_s, region_t))
    S_x_null.append(entanglement_entropy(W_s, region_x))

print(f"\n  Null (scrambled DAG):")
print(f"  Temporal partition: S = {np.mean(S_t_null):.3f} +/- {np.std(S_t_null):.3f}")
print(f"  Spatial partition:  S = {np.mean(S_x_null):.3f} +/- {np.std(S_x_null):.3f}")
null_ratio = np.mean(S_t_null) / max(np.mean(S_x_null), 1e-15)
print(f"  Null ratio S_t / S_x = {null_ratio:.3f}")
print(f"\n  Key question: is the temporal/spatial asymmetry in the causet-SJ vacuum")
print(f"  significantly different from the scrambled control?")
print(f"  Causet ratio: {ratio_ts:.3f}, Null ratio: {null_ratio:.3f}")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 81: SJ VACUUM 'FIDELITY' ACROSS MCMC TRAJECTORY")
print("How does the SJ vacuum change as the causet evolves under MCMC?")
print("Fidelity F(W1, W2) = Tr(sqrt(sqrt(W1)*W2*sqrt(W1)))")
print("If F stays high: vacuum is robust. If F drops: phase transition.")
print("=" * 75)

# Track how the SJ Wightman function changes step-by-step under MCMC.
# This probes the DYNAMICS of quantum geometry — something nobody has studied.
# At the BD phase transition, we expect a sharp drop in fidelity.

N = 40
n_mcmc_steps = 100

to = TwoOrder(N, rng=rng)
cs = to.to_causet()
W_prev, _, _ = sj_full(cs)

fidelities = []
of_list = [cs.ordering_fraction()]

for step in range(n_mcmc_steps):
    to_new = swap_move(to, rng)
    cs_new = to_new.to_causet()
    W_new, _, _ = sj_full(cs_new)

    # Simplified fidelity: use trace inner product
    # F = Tr(W1 * W2) / sqrt(Tr(W1^2) * Tr(W2^2))
    tr12 = np.trace(W_prev @ W_new)
    tr11 = np.trace(W_prev @ W_prev)
    tr22 = np.trace(W_new @ W_new)
    denom = np.sqrt(max(tr11 * tr22, 1e-30))
    F = tr12 / denom if denom > 0 else 0.0
    fidelities.append(F)
    of_list.append(cs_new.ordering_fraction())

    W_prev = W_new
    to = to_new

print(f"\n  Fidelity across {n_mcmc_steps} MCMC steps (N={N}):")
print(f"  Mean F = {np.mean(fidelities):.4f} +/- {np.std(fidelities):.4f}")
print(f"  Min F  = {np.min(fidelities):.4f}")
print(f"  Max F  = {np.max(fidelities):.4f}")
print(f"  Ordering fraction: {of_list[0]:.3f} -> {of_list[-1]:.3f}")

# What fraction of steps change W significantly?
n_low_fidelity = sum(1 for f in fidelities if f < 0.99)
print(f"  Steps with F < 0.99: {n_low_fidelity}/{n_mcmc_steps} ({100*n_low_fidelity/n_mcmc_steps:.0f}%)")
n_very_low = sum(1 for f in fidelities if f < 0.95)
print(f"  Steps with F < 0.95: {n_very_low}/{n_mcmc_steps} ({100*n_very_low/n_mcmc_steps:.0f}%)")

# Autocorrelation of fidelity
if len(fidelities) > 10:
    f_arr = np.array(fidelities)
    f_centered = f_arr - f_arr.mean()
    autocorr = np.correlate(f_centered, f_centered, 'full') / (f_arr.var() * len(f_arr))
    autocorr = autocorr[len(f_arr) - 1:]
    # Find first zero crossing
    zero_cross = np.where(autocorr < 0)[0]
    tau = zero_cross[0] if len(zero_cross) > 0 else len(autocorr)
    print(f"  Autocorrelation time for fidelity: tau ~ {tau} steps")
    print(f"  (This is the 'SJ vacuum decorrelation time' — a new observable)")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 82: 'CAUSAL ENTROPY' — ENTROPY OF THE CAUSAL MATRIX ITSELF")
print("Treat C as a probability matrix (normalize rows). Compute its entropy.")
print("This is a purely causal observable (no SJ vacuum needed).")
print("=" * 75)

# The causal matrix C[i,j] = 1 if i < j. Normalize each row to get a
# "transition probability" p_i(j) = C[i,j] / sum_j C[i,j].
# The entropy S_i = -sum_j p_i(j) ln(p_i(j)) measures the "uncertainty
# of the future" for element i. The distribution of {S_i} is a new
# invariant of the causal set.

Ns_82 = [50, 100, 150]

for N in Ns_82:
    # Sprinkled causet
    cs, _ = sprinkle_fast(N, dim=2, extent_t=1.0, rng=rng)
    C = cs.order.astype(float)
    row_sums = C.sum(axis=1)
    causal_entropies = []
    for i in range(N):
        if row_sums[i] > 0:
            p = C[i, :] / row_sums[i]
            p = p[p > 0]
            S = -np.sum(p * np.log(p))
            causal_entropies.append(S)

    # Random 2-order for comparison
    to = TwoOrder(N, rng=rng)
    cs_r = to.to_causet()
    C_r = cs_r.order.astype(float)
    row_sums_r = C_r.sum(axis=1)
    causal_ent_r = []
    for i in range(N):
        if row_sums_r[i] > 0:
            p = C_r[i, :] / row_sums_r[i]
            p = p[p > 0]
            S = -np.sum(p * np.log(p))
            causal_ent_r.append(S)

    # Random DAG
    cs_dag = random_dag(N, cs.ordering_fraction(), rng)
    C_d = cs_dag.order.astype(float)
    row_sums_d = C_d.sum(axis=1)
    causal_ent_d = []
    for i in range(N):
        if row_sums_d[i] > 0:
            p = C_d[i, :] / row_sums_d[i]
            p = p[p > 0]
            S = -np.sum(p * np.log(p))
            causal_ent_d.append(S)

    print(f"\n  N={N}:")
    print(f"    Sprinkled: <S_causal> = {np.mean(causal_entropies):.3f} +/- {np.std(causal_entropies):.3f}")
    print(f"    2-order:   <S_causal> = {np.mean(causal_ent_r):.3f} +/- {np.std(causal_ent_r):.3f}")
    print(f"    Random DAG:<S_causal> = {np.mean(causal_ent_d):.3f} +/- {np.std(causal_ent_d):.3f}")
    # KS test between sprinkled and random DAG
    ks, p_val = stats.ks_2samp(causal_entropies, causal_ent_d)
    print(f"    KS test (sprinkled vs DAG): D={ks:.3f}, p={p_val:.4f}")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 83: MUTUAL INFORMATION STRUCTURE — 'INFORMATION GEOMETRY'")
print("Build the mutual information matrix I(i,j) for ALL pairs.")
print("Is the MI matrix a metric? Does it encode the causal structure?")
print("=" * 75)

# Mutual information I(A:B) for individual elements as singleton regions
# is trivial. Instead, use "blocks": divide causet into k blocks,
# compute I(block_a : block_b) for all pairs. This gives a k x k matrix.
# Does the structure of this matrix differ between sprinkled and random?

N = 60
n_blocks = 6
block_size = N // n_blocks

# Sprinkled
cs, coords = sprinkle_fast(N, dim=2, extent_t=1.0, rng=rng)
W, _, _ = sj_full(cs)

# Partition elements by time coordinate into blocks
t_order = np.argsort(coords[:, 0])
blocks = [list(t_order[i * block_size:(i + 1) * block_size]) for i in range(n_blocks)]

MI_matrix = np.zeros((n_blocks, n_blocks))
for a in range(n_blocks):
    for b in range(a + 1, n_blocks):
        S_a = entanglement_entropy(W, blocks[a])
        S_b = entanglement_entropy(W, blocks[b])
        S_ab = entanglement_entropy(W, sorted(blocks[a] + blocks[b]))
        MI_matrix[a, b] = S_a + S_b - S_ab
        MI_matrix[b, a] = MI_matrix[a, b]

print(f"\n  MI matrix (sprinkled, time-ordered blocks):")
for a in range(n_blocks):
    row = [f"{MI_matrix[a, b]:.3f}" for b in range(n_blocks)]
    print(f"    [{', '.join(row)}]")

# Key property: does MI decay with "distance" (block separation)?
mi_by_sep = {}
for a in range(n_blocks):
    for b in range(a + 1, n_blocks):
        sep = b - a
        mi_by_sep.setdefault(sep, []).append(MI_matrix[a, b])

print(f"\n  MI vs block separation:")
for sep in sorted(mi_by_sep.keys()):
    vals = mi_by_sep[sep]
    print(f"    sep={sep}: <MI> = {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

# Does MI decay? Fit power law / exponential
seps = []
mis = []
for sep in sorted(mi_by_sep.keys()):
    seps.append(sep)
    mis.append(np.mean(mi_by_sep[sep]))

if len(seps) >= 3:
    log_seps = np.log(seps)
    log_mis = np.log(np.array(mis) + 1e-15)
    slope, intercept, r, p, se = stats.linregress(log_seps, log_mis)
    print(f"    Power law fit: MI ~ sep^{slope:.2f} (r={r:.3f})")
    print(f"    (For CFT: MI ~ 1/sep^{2:.0f})")

# Null: random partition
blocks_random = [list(rng.choice(N, size=block_size, replace=False)) for _ in range(n_blocks)]
MI_null = np.zeros((n_blocks, n_blocks))
for a in range(n_blocks):
    for b in range(a + 1, n_blocks):
        S_a = entanglement_entropy(W, blocks_random[a])
        S_b = entanglement_entropy(W, blocks_random[b])
        union = sorted(set(blocks_random[a]) | set(blocks_random[b]))
        S_ab = entanglement_entropy(W, union)
        MI_null[a, b] = S_a + S_b - S_ab

print(f"\n  MI for random partition (no geometric structure):")
mi_random_vals = MI_null[MI_null > 0]
if len(mi_random_vals) > 0:
    print(f"    Mean MI = {np.mean(mi_random_vals):.4f}")
    print(f"    Ratio ordered/random: {np.mean(list(mis)) / max(np.mean(mi_random_vals), 1e-15):.2f}")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 84: SPECTRAL GAP OF H = i*Delta AS ORDER PARAMETER")
print("The spectral gap of the SJ Hermitian may distinguish manifold phases.")
print("At the BD transition, does the gap close/open?")
print("=" * 75)

# The spectral gap = smallest positive eigenvalue of H = i*(2/N)(C^T - C).
# For a manifold-like causet, the gap should correspond to the IR scale
# (inversely proportional to the volume). For a non-manifold (crystal),
# the gap structure should be different.

# Test on a range of ordering fractions (tuned via 2-order structure)
# by varying the "squeezing" of 2-order coordinates.
Ns_84 = [50, 80, 120]
n_trials_84 = 5

for N in Ns_84:
    gaps_sprinkle = []
    gaps_random = []
    gaps_dag = []

    for trial in range(n_trials_84):
        # Sprinkled
        cs, _ = sprinkle_fast(N, dim=2, extent_t=1.0, rng=rng)
        C = cs.order.astype(float)
        H = 1j * (2.0 / N) * (C.T - C)
        evals = np.linalg.eigvalsh(H).real
        pos = sorted(evals[evals > 1e-12])
        if len(pos) >= 1:
            gaps_sprinkle.append(pos[0])

        # Random 2-order
        to = TwoOrder(N, rng=rng)
        cs_r = to.to_causet()
        C_r = cs_r.order.astype(float)
        H_r = 1j * (2.0 / N) * (C_r.T - C_r)
        evals_r = np.linalg.eigvalsh(H_r).real
        pos_r = sorted(evals_r[evals_r > 1e-12])
        if len(pos_r) >= 1:
            gaps_random.append(pos_r[0])

        # Random DAG
        cs_d = random_dag(N, cs.ordering_fraction(), rng)
        C_d = cs_d.order.astype(float)
        H_d = 1j * (2.0 / N) * (C_d.T - C_d)
        evals_d = np.linalg.eigvalsh(H_d).real
        pos_d = sorted(evals_d[evals_d > 1e-12])
        if len(pos_d) >= 1:
            gaps_dag.append(pos_d[0])

    print(f"\n  N={N}:")
    if gaps_sprinkle:
        print(f"    Sprinkled: gap = {np.mean(gaps_sprinkle):.5f} +/- {np.std(gaps_sprinkle):.5f}")
    if gaps_random:
        print(f"    2-order:   gap = {np.mean(gaps_random):.5f} +/- {np.std(gaps_random):.5f}")
    if gaps_dag:
        print(f"    Random DAG:gap = {np.mean(gaps_dag):.5f} +/- {np.std(gaps_dag):.5f}")

    # Scaling: gap ~ N^(-alpha)? Check for sprinkled
    if gaps_sprinkle:
        print(f"    gap * N = {np.mean(gaps_sprinkle) * N:.4f} (const if gap ~ 1/N)")
        print(f"    gap * N^(2/d) = {np.mean(gaps_sprinkle) * N:.4f} (d=2)")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 85: TOPOLOGICAL INVARIANT — BETTI NUMBERS OF THE ORDER COMPLEX")
print("The order complex of a causal set has topological invariants.")
print("For a causal diamond: should be contractible (Betti_0=1, rest=0).")
print("Does this hold? Do random DAGs differ?")
print("=" * 75)

# The order complex of a poset P: simplices are the chains (totally ordered subsets).
# For a causal diamond in d-dim Minkowski: contractible → trivial homology.
# For a random DAG: likely different topology.
#
# Computing full homology is expensive, but we can compute:
# - Betti_0 = number of connected components of the comparability graph
# - Euler characteristic via Mobius function
# - "Topological charge" from the link matrix

N = 60
n_trials_85 = 5

def comparability_components(cs):
    """Number of connected components of the comparability graph."""
    N = cs.n
    adj = cs.order | cs.order.T  # i and j are comparable if i<j or j<i
    # BFS to find components
    visited = set()
    n_comp = 0
    for start in range(N):
        if start in visited:
            continue
        n_comp += 1
        queue = [start]
        visited.add(start)
        while queue:
            node = queue.pop(0)
            for k in range(N):
                if adj[node, k] and k not in visited:
                    visited.add(k)
                    queue.append(k)
    return n_comp

def euler_characteristic_links(cs):
    """Euler characteristic from the link graph (Hasse diagram).
    chi = V - E where V = nodes, E = links."""
    N = cs.n
    links = cs.link_matrix()
    n_links = int(np.sum(links))  # directed links
    return N - n_links

print(f"\n  Topological invariants (N={N}):")
print(f"  {'Type':<15} {'Betti_0':<10} {'chi(links)':<12} {'links/N':<10}")

for trial in range(n_trials_85):
    # Sprinkled
    cs, _ = sprinkle_fast(N, dim=2, extent_t=1.0, rng=rng)
    b0 = comparability_components(cs)
    chi = euler_characteristic_links(cs)
    links_per_n = np.sum(cs.link_matrix()) / N

    # Random 2-order
    to = TwoOrder(N, rng=rng)
    cs_r = to.to_causet()
    b0_r = comparability_components(cs_r)
    chi_r = euler_characteristic_links(cs_r)
    links_per_n_r = np.sum(cs_r.link_matrix()) / N

    # Random DAG
    cs_d = random_dag(N, cs.ordering_fraction(), rng)
    b0_d = comparability_components(cs_d)
    chi_d = euler_characteristic_links(cs_d)
    links_per_n_d = np.sum(cs_d.link_matrix()) / N

    if trial == 0:
        print(f"  {'Sprinkled':<15} {b0:<10} {chi:<12} {links_per_n:<10.2f}")
        print(f"  {'2-order':<15} {b0_r:<10} {chi_r:<12} {links_per_n_r:<10.2f}")
        print(f"  {'Random DAG':<15} {b0_d:<10} {chi_d:<12} {links_per_n_d:<10.2f}")

# Now do a systematic comparison: chi / N for different types
chi_sprinkle = []
chi_random = []
chi_dag = []

for trial in range(n_trials_85):
    cs, _ = sprinkle_fast(N, dim=2, extent_t=1.0, rng=rng)
    chi_sprinkle.append(euler_characteristic_links(cs) / N)

    to = TwoOrder(N, rng=rng)
    cs_r = to.to_causet()
    chi_random.append(euler_characteristic_links(cs_r) / N)

    cs_d = random_dag(N, cs.ordering_fraction(), rng)
    chi_dag.append(euler_characteristic_links(cs_d) / N)

print(f"\n  chi/N statistics:")
print(f"    Sprinkled: {np.mean(chi_sprinkle):.3f} +/- {np.std(chi_sprinkle):.3f}")
print(f"    2-order:   {np.mean(chi_random):.3f} +/- {np.std(chi_random):.3f}")
print(f"    Random DAG:{np.mean(chi_dag):.3f} +/- {np.std(chi_dag):.3f}")

# KS test: can chi/N distinguish sprinkled from DAG?
ks, p_val = stats.ks_2samp(chi_sprinkle, chi_dag)
print(f"    KS (sprinkled vs DAG): D={ks:.3f}, p={p_val:.4f}")


# ================================================================
# FINAL SUMMARY AND SCORES
# ================================================================
print("\n" + "=" * 75)
print("FINAL SCORES")
print("=" * 75)
print("""
Scoring criteria:
  8+ requires: genuinely novel, passes null controls, connects to known physics,
  would be interesting to QG community, survives finite-size scrutiny.

IDEA 76 — SJ Vacuum: Sprinkled vs 2-orders
  c_eff: sprinkled ~2.89-3.35, random ~3.06-3.50 (sprinkled ~5% lower)
  Ordering fractions nearly identical (~0.50), so density isn't the cause.
  Sprinkled causets have LOWER c_eff — the geometry constrains entanglement.
  But both diverge with N (neither approaches c=1). The difference is small.
  SCORE: 5/10 — Interesting observation but the effect is small and both
  still diverge. Not enough for a paper alone. Would need MCMC continuum phase.

IDEA 77 — SJ weight vs volume of past
  r = 0.931 for causet W, but r = 0.978 for RANDOM W (!).
  The null model actually has HIGHER correlation. This makes sense: random W
  elements are all positive, so more elements in past = bigger sum trivially.
  The SJ W has signs, which REDUCE the correlation.
  SCORE: 2/10 — Null model kills it. The correlation is just counting.

IDEA 78 — Causal matrix spectrum vs SJ spectrum
  C has all-zero eigenvalues (upper triangular). Not interesting.
  C + C^T has broad spectrum. SJ H has ~47% positive eigenvalues.
  Random DAG has ~46% positive — nearly identical.
  Fraction of positive eigenvalues is NOT geometry-dependent.
  SCORE: 3/10 — No discrimination between causet and random DAG.

IDEA 79 — W(i,j) vs causal distance
  Power law |W| ~ d^{-0.61} with Delta = 0.30.
  But r = -0.654 with p = 0.23 — NOT statistically significant.
  Too few data points at each distance bin. Would need much larger N.
  The 4-point function is trivially zero (Gaussian by construction).
  SCORE: 4/10 — The distance-dependence direction is interesting
  but data is too noisy. The "non-Gaussianity" angle was a dead end
  (state is Gaussian by construction).

IDEA 80 — Timelike vs spacelike entanglement asymmetry
  S_temporal / S_spatial = 1.001 for sprinkled causets.
  Null ratio = 0.993. Essentially NO asymmetry.
  The SJ vacuum treats temporal and spatial partitions identically.
  This makes physical sense: the SJ vacuum is Lorentz-invariant,
  so there's no preferred frame to distinguish t from x.
  SCORE: 3/10 — Clean null result. The Lorentz invariance of the SJ
  vacuum means t/x symmetry is expected. Not publishable.

IDEA 81 — SJ vacuum fidelity under MCMC
  Mean F = 0.97, with 69% of steps having F < 0.99.
  Autocorrelation time tau ~ 1 step.
  This means each MCMC move changes W significantly — the vacuum is
  NOT robust under single coordinate swaps. This is expected: at N=40,
  each element is ~2.5% of the causet, so swapping one coordinate
  changes ~5% of causal relations.
  SCORE: 4/10 — Observable is well-defined but results are unsurprising.
  Would need to study this at the BD phase transition (F should show
  critical slowing down) to make it interesting. Not enough alone.

IDEA 82 — Causal entropy (row entropy of C)
  Sprinkled vs DAG: KS p < 0.003 at all N tested! Strong discrimination.
  Sprinkled causets have LOWER causal entropy than random DAGs.
  Sprinkled and 2-orders have similar causal entropy.
  The difference grows with N: 2.19 vs 2.97 (N=50), 3.18 vs 3.99 (N=150).
  This makes physical sense: in a manifold, the future light cone constrains
  which elements can be in the future. Random DAGs have no such constraint.
  SCORE: 6/10 — Genuinely discriminates manifold-like from random. But it's
  essentially measuring that manifold causets have more structured causal
  relations (which the ordering fraction already captures). The entropy
  formulation is novel but the underlying signal may not be new. Would need
  to show it captures something BEYOND ordering fraction.

IDEA 83 — Mutual information geometry
  MI decays with block separation: ~sep^{-0.27}.
  CFT predicts MI ~ sep^{-2}. Our decay is much slower.
  MI for geometric partitions is LOWER than for random partitions (0.10 vs 0.39).
  This is a bit surprising: geometric neighbors should be more correlated.
  But the time-ordered blocks span the full spatial extent, so adjacent
  time blocks are causally well-connected → high MI expected.
  Non-monotonic behavior at sep=5 (wrapping effects from diamond geometry).
  SCORE: 5/10 — The MI decay power law is interesting but the exponent
  doesn't match CFT predictions, and the non-monotonic behavior at large
  separation suggests finite-size/geometry artifacts. Needs larger N and
  spatial (not temporal) partitioning.

IDEA 84 — Spectral gap as order parameter
  gap * N is NOT constant: 0.22 → 0.15 → 0.08 as N grows.
  So gap ~ N^{-alpha} with alpha > 1. This means gap closes FASTER than 1/N.
  Fit: gap ~ N^{-1.1} approximately.
  No significant difference between sprinkled, 2-order, and random DAG.
  SCORE: 3/10 — The gap scaling is a valid measurement but it doesn't
  distinguish causets from random DAGs. Not geometry-dependent.

IDEA 85 — Topological invariants (Euler characteristic)
  chi/N: sprinkled = -1.79, 2-order = -1.81, random DAG = -0.67.
  KS test: D = 1.0, p = 0.008 — PERFECT separation!
  Sprinkled and 2-orders have the SAME chi/N (both ~-1.8).
  Random DAGs have much less negative chi/N (~-0.7).
  This is because manifold causets have more links per element (2.5 vs 1.35).
  chi = N - L, so more links → more negative chi.
  SCORE: 6/10 — Clean discrimination with strong statistical significance.
  But chi/N is essentially measuring links/N (same as link density).
  The topological framing is novel but the underlying signal is known.
  To elevate: need higher Betti numbers or the actual order complex homology.

SUMMARY: No 8+ found in this round.
Best ideas: #82 (causal entropy, 6/10) and #85 (Euler characteristic, 6/10).
Both discriminate manifold from random but are essentially measuring link density
in different guises. The SJ vacuum ideas (#76, #80, #81, #83) all produced
interesting observations but either failed null tests or showed effects too
small for a standalone paper.

The most PROMISING direction for follow-up:
- Idea 76 (sprinkled vs 2-order c_eff) at the BD continuum phase
- Idea 82 (causal entropy) if it can be shown to capture MORE than link density
- Idea 81 (SJ fidelity) specifically at the BD critical point
""")

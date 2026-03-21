"""
Experiment 46: Moonshots Round 2 — Thinking Differently

Ideas 11-20. More creative, less conservative.
Focus on things that would be genuinely surprising.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from causal_sets.two_orders import TwoOrder
from causal_sets.two_orders_v2 import mcmc_corrected, bd_action_corrected
from causal_sets.fast_core import FastCausalSet
from causal_sets.bd_action import count_intervals_by_size

rng = np.random.default_rng(42)


def sj_full(cs):
    """Return W and the full eigendecomposition."""
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


# ================================================================
print("=" * 75)
print("IDEA 11: ENTANGLEMENT METRIC")
print("Define distance from W. Does it recover the embedding metric?")
print("=" * 75)

# Distance between elements i and j defined as:
# d_ent(i,j) = -ln(|W[i,j]| / sqrt(W[i,i] * W[j,j]))
# This is like a "correlation distance" — high correlation = short distance.

N = 50
to = TwoOrder(N, rng=rng)
cs = to.to_causet()
W, _, _ = sj_full(cs)

# True Minkowski distance: d_Mink = sqrt(|dt^2 - dx^2|)
# In 2-order coordinates: u = to.u, v = to.v
# Minkowski coords: t = (u+v)/2, x = (v-u)/2
t_true = (to.u + to.v) / (2 * N)
x_true = (to.v - to.u) / (2 * N)

# Entanglement distance
d_ent_list = []
d_mink_list = []
for i in range(N):
    for j in range(i + 1, N):
        w_ij = abs(W[i, j])
        w_ii = max(abs(W[i, i]), 1e-15)
        w_jj = max(abs(W[j, j]), 1e-15)
        if w_ij > 1e-15:
            d_ent = -np.log(w_ij / np.sqrt(w_ii * w_jj))
        else:
            d_ent = 20.0  # large distance for uncorrelated pairs

        dt = t_true[j] - t_true[i]
        dx = x_true[j] - x_true[i]
        d_mink = np.sqrt(abs(dt ** 2 - dx ** 2))

        d_ent_list.append(d_ent)
        d_mink_list.append(d_mink)

r_metric = np.corrcoef(d_ent_list, d_mink_list)[0, 1]
print(f"\n  Correlation between d_ent and d_Minkowski: r = {r_metric:.3f}")
print(f"  If r > 0.5: entanglement defines a metric consistent with Minkowski")

# Null test: random W
W_rand = np.abs(rng.standard_normal((N, N))) * 0.01
W_rand = (W_rand + W_rand.T) / 2
np.fill_diagonal(W_rand, 0.3)
d_rand = []
for i in range(N):
    for j in range(i + 1, N):
        w_ij = abs(W_rand[i, j])
        w_ii = max(abs(W_rand[i, i]), 1e-15)
        w_jj = max(abs(W_rand[j, j]), 1e-15)
        d_rand.append(-np.log(max(w_ij, 1e-15) / np.sqrt(w_ii * w_jj)))

r_null = np.corrcoef(d_rand, d_mink_list)[0, 1]
print(f"  Null (random W): r = {r_null:.3f}")
print(f"  z-score: {(r_metric - r_null) / 0.05:.1f}")

# ================================================================
print("\n" + "=" * 75)
print("IDEA 12: MODULAR HAMILTONIAN = BOOST GENERATOR?")
print("Does the modular flow match Rindler time evolution?")
print("=" * 75)

# The modular Hamiltonian K_A = -ln(rho_A) generates modular flow.
# In Rindler space, K is proportional to the boost generator.
# On a causal set, check: does K_A have eigenvectors that align with
# the boost direction (t → t*cosh(s) + x*sinh(s))?

N = 50
to = TwoOrder(N, rng=rng)
cs = to.to_causet()
W, _, _ = sj_full(cs)

A = list(range(N // 2))
W_A = W[np.ix_(A, A)]
eigs_A = np.linalg.eigvalsh(W_A)
eigs_A = np.clip(eigs_A, 1e-15, 1 - 1e-15)

# Modular Hamiltonian eigenvalues: -ln(nu / (1-nu))
K_evals = -np.log(eigs_A / (1 - eigs_A))

# The boost generator in Rindler coordinates would give equally spaced
# K eigenvalues (thermal spectrum). Check spacing.
K_sorted = np.sort(K_evals)
spacings = np.diff(K_sorted)
cv = np.std(spacings) / (np.mean(spacings) + 1e-10)
print(f"\n  Modular Hamiltonian eigenvalue spacings:")
print(f"    Mean spacing: {np.mean(spacings):.3f}")
print(f"    CV (0=thermal): {cv:.3f}")
print(f"    {'THERMAL (boost-like)' if cv < 0.3 else 'NOT thermal'}")

# ================================================================
print("\n" + "=" * 75)
print("IDEA 13: INFORMATION PROPAGATION SPEED")
print("Is there a finite speed of entanglement propagation?")
print("=" * 75)

# Perturb element 0 (add/remove a relation). Track how S(region) changes
# for regions at increasing causal distance from element 0.

N = 50
to = TwoOrder(N, rng=rng)
cs = to.to_causet()
W0, _, _ = sj_full(cs)

# Compute baseline entropy of small regions around each element
baseline_S = {}
for x in range(0, N, 5):
    region = [j for j in range(N) if abs(j - x) < 5]
    baseline_S[x] = entanglement_entropy(W0, region)

# Perturb: flip a relation involving element 0
cs_pert = FastCausalSet(N)
cs_pert.order = cs.order.copy()
# Find a relation from element 0
targets = [j for j in range(N) if cs.order[0, j]]
if targets:
    cs_pert.order[0, targets[0]] = False  # remove one relation

W_pert, _, _ = sj_full(cs_pert)

print(f"\n  Perturbation: remove one relation from element 0")
print(f"  {'distance':>10} {'|delta_S|':>10}")
print("-" * 25)

for x in sorted(baseline_S.keys()):
    region = [j for j in range(N) if abs(j - x) < 5]
    S_pert = entanglement_entropy(W_pert, region)
    delta_S = abs(S_pert - baseline_S[x])
    print(f"  {x:>10} {delta_S:>10.4f}")

print(f"  If |delta_S| decays with distance: finite propagation speed")
print(f"  If |delta_S| is uniform: instantaneous propagation")

# ================================================================
print("\n" + "=" * 75)
print("IDEA 14: ENTANGLEMENT CONTOUR")
print("Per-element contribution to S(A) — does it match CFT?")
print("=" * 75)

# The entanglement contour s(x) assigns to each element x in A
# its contribution to S(A), such that S(A) = sum_x s(x).
# In a 1+1D CFT: s(x) = (c/6) * 1/(x * (L-x)) * L  (for an interval [0,L])
# This peaks at the boundary of A and is minimal in the interior.

N = 60
to = TwoOrder(N, rng=rng)
cs = to.to_causet()
W, _, _ = sj_full(cs)

A = list(range(N // 2))
S_total = entanglement_entropy(W, A)

# Compute contour: s(x) = S(A) - S(A\{x}) approximately
contour = []
for idx, x in enumerate(A):
    A_minus_x = [a for a in A if a != x]
    S_minus = entanglement_entropy(W, A_minus_x)
    s_x = S_total - S_minus  # marginal contribution of element x
    contour.append(s_x)

contour = np.array(contour)

# CFT prediction: s(x) peaks at boundaries (x near 0 and x near N/2)
print(f"\n  Entanglement contour s(x) for A = [0, N/2):")
print(f"  {'position':>10} {'s(x)':>8}")
print("-" * 22)
for idx in range(0, len(A), max(1, len(A) // 10)):
    print(f"  {idx:>10} {contour[idx]:>8.4f}")

# Does it peak at the boundary?
boundary_s = (contour[0] + contour[-1]) / 2
interior_s = np.mean(contour[len(A) // 4: 3 * len(A) // 4])
print(f"\n  Boundary s: {boundary_s:.4f}")
print(f"  Interior s: {interior_s:.4f}")
print(f"  Ratio boundary/interior: {boundary_s / interior_s:.2f}")
print(f"  CFT predicts: ratio >> 1 (contour peaks at boundary)")
print(f"  Volume law: ratio = 1 (uniform contour)")

# ================================================================
print("\n" + "=" * 75)
print("IDEA 15: DIMENSION FROM THE SPECTRAL GAP")
print("Does gap × N^(1-2/d) give a universal constant?")
print("=" * 75)

from causal_sets.d_orders import DOrder

print(f"\n  If gap ~ N^(2/d - 1), then gap × N^(1-2/d) = const for each d")
print(f"  {'d':>3} {'N':>5} {'gap':>10} {'gap*N':>10} {'gap*N^(1-2/d)':>14}")
print("-" * 48)

for d in [2, 3, 4, 5]:
    for N_test in [20, 30, 40]:
        gaps = []
        for _ in range(12):
            if d == 2:
                to = TwoOrder(N_test, rng=rng)
                cs_d = to.to_causet()
            else:
                do = DOrder(d=d, N=N_test, rng=rng)
                cs_d = do.to_causet_fast()

            C = cs_d.order.astype(float)
            Delta = 0.5 * (C.T - C)
            H = 1j * Delta
            evals = np.linalg.eigvalsh(H).real
            pos = evals[evals > 1e-10]
            if len(pos) > 0:
                gaps.append(pos.min())

        if gaps:
            g = np.mean(gaps)
            exponent = 1 - 2.0 / d
            scaled = g * N_test ** exponent if exponent != 0 else g * np.log(N_test)
            print(f"  {d:>3} {N_test:>5} {g:>10.4f} {g * N_test:>10.2f} {scaled:>14.3f}")
    print()

# ================================================================
print("=" * 75)
print("IDEA 16: TWO CAUSETS CONNECTED BY ENTANGLEMENT")
print("Entangle two separate causets. Does geometry emerge between them?")
print("=" * 75)

# Create two separate causal sets A and B (no causal relations between them).
# Entangle their SJ vacua by constructing a joint state.
# Measure: does mutual information I(A:B) create an effective
# "causal connection" that wasn't there before?

N_each = 20
# Two separate 2-orders
to_A = TwoOrder(N_each, rng=rng)
cs_A = to_A.to_causet()
to_B = TwoOrder(N_each, rng=rng)
cs_B = to_B.to_causet()

# Combined causal set: no relations between A and B
N_total = 2 * N_each
cs_combined = FastCausalSet(N_total)
# Copy A's relations
for i in range(N_each):
    for j in range(N_each):
        cs_combined.order[i, j] = cs_A.order[i, j]
# Copy B's relations (shifted by N_each)
for i in range(N_each):
    for j in range(N_each):
        cs_combined.order[N_each + i, N_each + j] = cs_B.order[i, j]

W_combined, _, _ = sj_full(cs_combined)

# Measure MI between A and B
region_A = list(range(N_each))
region_B = list(range(N_each, N_total))
I_AB = (entanglement_entropy(W_combined, region_A) +
        entanglement_entropy(W_combined, region_B) -
        entanglement_entropy(W_combined, region_A + region_B))

# Cross-correlations
w_cross = []
for i in range(N_each):
    for j in range(N_each, N_total):
        w_cross.append(abs(W_combined[i, j]))

print(f"\n  Two disconnected causets (N={N_each} each):")
print(f"    I(A:B) = {I_AB:.6f}")
print(f"    Mean |W| cross-correlation: {np.mean(w_cross):.6f}")
print(f"    Max |W| cross-correlation: {np.max(w_cross):.6f}")
if I_AB > 0.01:
    print(f"    ENTANGLEMENT EXISTS between disconnected causets!")
    print(f"    This is 'entanglement creates geometry' — ER=EPR in reverse")
else:
    print(f"    No significant entanglement (expected for disconnected systems)")

# Now ADD a few cross-relations and see if MI increases
for n_cross in [1, 3, 5, 10]:
    cs_linked = FastCausalSet(N_total)
    cs_linked.order = cs_combined.order.copy()
    # Add random cross-relations
    for _ in range(n_cross):
        i = rng.integers(0, N_each)
        j = rng.integers(N_each, N_total)
        cs_linked.order[i, j] = True

    W_linked, _, _ = sj_full(cs_linked)
    I_linked = (entanglement_entropy(W_linked, region_A) +
                entanglement_entropy(W_linked, region_B) -
                entanglement_entropy(W_linked, region_A + region_B))

    print(f"    {n_cross} cross-relations added: I(A:B) = {I_linked:.4f}")

print(f"  If I grows with cross-relations: geometry creates entanglement (standard)")
print(f"  The reverse (entanglement creates geometry) would require starting")
print(f"  with entangled but disconnected systems and seeing emergent geometry")

# ================================================================
print("\n" + "=" * 75)
print("IDEA 17: SVD OF W — DO SINGULAR VECTORS = LEFT/RIGHT MOVERS?")
print("=" * 75)

N = 50
to = TwoOrder(N, rng=rng)
cs = to.to_causet()
W, _, _ = sj_full(cs)

U_svd, s_svd, Vt_svd = np.linalg.svd(W)

# In a 2D CFT, the field decomposes into left-movers (u direction)
# and right-movers (v direction). Do the SVD vectors align with
# the u and v coordinates?
u_coords = to.u / N
v_coords = to.v / N

# Top singular vector correlations with coordinates
for k in range(3):
    r_u = abs(np.corrcoef(U_svd[:, k], u_coords)[0, 1])
    r_v = abs(np.corrcoef(U_svd[:, k], v_coords)[0, 1])
    print(f"  SVD vector {k}: |r(U_k, u)| = {r_u:.3f}, |r(U_k, v)| = {r_v:.3f}")

print(f"  If U vectors align with u OR v: left/right mover decomposition")

# ================================================================
print("\n" + "=" * 75)
print("IDEA 18: CAUSAL SET DETERMINES SJ VACUUM — HOW UNIQUELY?")
print("Two different causets with same MM dimension — same S_ent?")
print("=" * 75)

# If two causets have the same dimension (by MM estimator) but different
# causal structure, do they have the same entanglement entropy?
# If yes: S is determined by dimension alone (boring).
# If no: S captures more than just dimension (interesting — it's sensitive
# to the detailed geometry, not just the topology).

from causal_sets.dimension import myrheim_meyer
from causal_sets.core import CausalSet

N = 40
# Generate 2-orders and 3-orders with matched MM dimension
results_2d = []
results_3d = []

for _ in range(20):
    # 2-order (true dimension 2)
    to = TwoOrder(N, rng=rng)
    cs2 = to.to_causet()
    W2, _, _ = sj_full(cs2)
    cs_old = CausalSet(N)
    cs_old.order = cs2.order.astype(np.int8)
    mm2 = myrheim_meyer(cs_old)
    S2 = entanglement_entropy(W2, list(range(N // 2)))
    results_2d.append((mm2, S2))

for _ in range(20):
    # 3-order (true dimension 3)
    do = DOrder(d=3, N=N, rng=rng)
    cs3 = do.to_causet_fast()
    W3, _, _ = sj_full(cs3)
    cs_old = CausalSet(N)
    cs_old.order = cs3.order.astype(np.int8)
    mm3 = myrheim_meyer(cs_old)
    S3 = entanglement_entropy(W3, list(range(N // 2)))
    results_3d.append((mm3, S3))

mm_2d = np.mean([r[0] for r in results_2d])
mm_3d = np.mean([r[0] for r in results_3d])
S_2d = np.mean([r[1] for r in results_2d])
S_3d = np.mean([r[1] for r in results_3d])

print(f"\n  2-orders: MM dim = {mm_2d:.2f}, S(N/2) = {S_2d:.3f}")
print(f"  3-orders: MM dim = {mm_3d:.2f}, S(N/2) = {S_3d:.3f}")
print(f"  Same S despite different dimension? {'YES (S is dimension-blind)' if abs(S_2d - S_3d) < 0.3 else 'NO (S distinguishes dimensions)'}")

# ================================================================
print("\n" + "=" * 75)
print("IDEA 19: HOLOGRAPHIC ENTROPY CONE")
print("Does the SJ vacuum satisfy ALL holographic entropy inequalities?")
print("=" * 75)

# Beyond monogamy (I₃ ≤ 0), holographic states satisfy a full set
# of entropy inequalities known as the holographic entropy cone.
# For 4 parties: there are additional inequalities beyond SSA and monogamy.
# The most notable: the cyclic inequality
# S(AB) + S(BC) + S(CD) + S(DA) ≥ S(ABCD) + S(A) + S(B) + S(C) + S(D) + S(AC) + S(BD)
# ... this is complex. Let me test a simpler one:
# MMI (monogamy of mutual information): I(A:BC) ≥ I(A:B) + I(A:C)

N = 40
to = TwoOrder(N, rng=rng)
cs = to.to_causet()
W, _, _ = sj_full(cs)

def mutual_info(W, A, B):
    return (entanglement_entropy(W, A) + entanglement_entropy(W, B)
            - entanglement_entropy(W, list(set(A) | set(B))))

# 4-party test: divide into quarters
quarter = N // 4
A = list(range(quarter))
B = list(range(quarter, 2 * quarter))
C = list(range(2 * quarter, 3 * quarter))
D = list(range(3 * quarter, N))

# Test several holographic inequalities
I_A_B = mutual_info(W, A, B)
I_A_C = mutual_info(W, A, C)
I_A_BC = mutual_info(W, A, list(set(B) | set(C)))

# MMI: I(A:BC) >= I(A:B) + I(A:C) (equivalent to I₃ ≤ 0)
mmi = I_A_BC - I_A_B - I_A_C
print(f"\n  4-party test (quarters):")
print(f"    I(A:B) = {I_A_B:.4f}")
print(f"    I(A:C) = {I_A_C:.4f}")
print(f"    I(A:BC) = {I_A_BC:.4f}")
print(f"    MMI = I(A:BC) - I(A:B) - I(A:C) = {mmi:.4f}")
print(f"    Holographic: MMI ≥ 0. Result: {'SATISFIED' if mmi >= -0.001 else 'VIOLATED'}")

# Test over many realizations
n_mmi_sat = 0
n_total = 30
for _ in range(n_total):
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    W, _, _ = sj_full(cs)

    perm = rng.permutation(N)
    q = N // 4
    A = list(perm[:q])
    B = list(perm[q:2*q])
    C = list(perm[2*q:3*q])

    I_AB = mutual_info(W, A, B)
    I_AC = mutual_info(W, A, C)
    I_ABC = mutual_info(W, A, list(set(B)|set(C)))

    if I_ABC >= I_AB + I_AC - 0.001:
        n_mmi_sat += 1

print(f"\n  MMI satisfied in {n_mmi_sat}/{n_total} random 4-partitions ({100*n_mmi_sat/n_total:.0f}%)")
print(f"  (This is equivalent to I₃ ≤ 0 monogamy — cross-check)")

# ================================================================
print("\n" + "=" * 75)
print("IDEA 20: ENTANGLEMENT BETWEEN CAUSAL AND SPACELIKE DEGREES OF FREEDOM")
print("Partition into 'timelike-connected' vs 'spacelike-connected' elements")
print("=" * 75)

# For a given element x, partition the rest into:
# T(x) = elements causally related to x (past ∪ future)
# S(x) = elements spacelike to x
# Compute I(T(x) : S(x)). In classical physics: I = 0.
# In quantum gravity: I > 0 would mean quantum correlations between
# causally connected and disconnected degrees of freedom.

N = 50
to = TwoOrder(N, rng=rng)
cs = to.to_causet()
W, _, _ = sj_full(cs)

I_TS_vals = []
for x in range(0, N, 5):
    T_x = [j for j in range(N) if j != x and (cs.order[x, j] or cs.order[j, x])]
    S_x = [j for j in range(N) if j != x and not cs.order[x, j] and not cs.order[j, x]]

    if len(T_x) < 2 or len(S_x) < 2:
        continue

    I = mutual_info(W, T_x, S_x)
    I_TS_vals.append(I)

print(f"\n  I(timelike : spacelike) for individual elements:")
print(f"    Mean: {np.mean(I_TS_vals):.4f}")
print(f"    Std: {np.std(I_TS_vals):.4f}")
print(f"    Max: {np.max(I_TS_vals):.4f}")
if np.mean(I_TS_vals) > 0.01:
    print(f"    QUANTUM correlations between causal and spacelike degrees of freedom!")
    print(f"    This means the SJ vacuum entangles the 'time' and 'space' directions")
else:
    print(f"    No significant time-space entanglement")

print("\n" + "=" * 75)
print("ALL 10 IDEAS (11-20) TESTED")
print("=" * 75)

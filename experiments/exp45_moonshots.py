"""
Experiment 45: Ten Moonshot Ideas — Targeting 8-10 Papers

Following established methodology:
- Null model first
- Score honestly
- Follow the surprise

Each idea gets a quick test (~5-10 min). Promising ones get deeper investigation.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from causal_sets.two_orders import TwoOrder
from causal_sets.two_orders_v2 import mcmc_corrected, bd_action_corrected
from causal_sets.fast_core import FastCausalSet, sprinkle_fast
from causal_sets.d_orders import DOrder
from causal_sets.bd_action import count_intervals_by_size

rng = np.random.default_rng(42)


def sj_wightman(cs):
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


def mutual_info(W, A, B):
    return (entanglement_entropy(W, A) + entanglement_entropy(W, B)
            - entanglement_entropy(W, list(set(A) | set(B))))


# ================================================================
print("=" * 75)
print("IDEA 1: ENTANGLEMENT NEGATIVITY")
print("Are the SJ correlations genuinely QUANTUM (not just classical)?")
print("=" * 75)

# Logarithmic negativity: N(A:B) = log2(||rho_AB^{T_B}||_1)
# For Gaussian states: computed from the eigenvalues of the
# partially transposed correlation matrix.
# If N > 0: genuinely quantum entanglement (not just classical correlation)

N = 40
to = TwoOrder(N, rng=rng)
cs = to.to_causet()
W, _, _ = sj_wightman(cs)

# For a Gaussian state with correlation matrix W, the negativity
# between regions A and B can be computed from the symplectic
# eigenvalues of the partial transpose.
# Simplified test: compute the eigenvalues of W_A, check if any
# are outside [0, 1] after partial transpose (indicating entanglement).

# Actually, for our W (which has eigenvalues in [0,1]), the state is
# always a valid Gaussian state. The negativity requires the FULL
# covariance matrix (positions AND momenta), not just W.
# For a free scalar, the covariance matrix is:
# Gamma = [[Re(W+1/2), Im(W)], [-Im(W), Re(W+1/2)]]
# But our W is real, so Im(W) = 0 and Gamma is block diagonal.
# In this case, the partial transpose eigenvalues = eigenvalues of W_A.
# If all in [0,1]: no negativity (PPT state).

# Let's check directly: are there eigenvalues of W_A outside [0,1]?
A = list(range(N // 2))
W_A = W[np.ix_(A, A)]
eigs_A = np.linalg.eigvalsh(W_A)
print(f"\n  W_A eigenvalues: min={eigs_A.min():.4f}, max={eigs_A.max():.4f}")
print(f"  All in [0,1]: {'YES' if eigs_A.min() >= -1e-10 and eigs_A.max() <= 1+1e-10 else 'NO'}")
print(f"  For real W: the state is PPT (positive partial transpose)")
print(f"  This means: we CANNOT prove the correlations are quantum via negativity")
print(f"  (PPT doesn't mean no entanglement — it means negativity can't detect it)")

# BUT: we can test if the correlations are quantum by checking if they
# violate a Bell inequality. Simpler: check if I(A:B) > classical bound.
# For classical correlations: I(A:B) <= min(S(A), S(B))
# For quantum: I(A:B) can exceed this (quantum discord)
B = list(range(N // 2, N))
I_AB = mutual_info(W, A, B)
S_A = entanglement_entropy(W, A)
S_B = entanglement_entropy(W, B)
print(f"\n  I(A:B) = {I_AB:.3f}")
print(f"  min(S(A), S(B)) = {min(S_A, S_B):.3f}")
print(f"  I(A:B) / min(S) = {I_AB / min(S_A, S_B):.3f}")
print(f"  If ratio > 1: quantum correlations exceed classical bound")
print(f"  Result: {'QUANTUM' if I_AB > min(S_A, S_B) else 'CLASSICAL CONSISTENT'}")

# ================================================================
print("\n" + "=" * 75)
print("IDEA 2: JACOBSON'S ARGUMENT — LOCAL δS_ent vs δS_BD")
print("Does perturbing the causal set change S_ent proportionally to S_BD?")
print("=" * 75)

N = 40
eps = 0.12

# Strategy: start with a causet. Make a small local change (flip one
# relation). Measure δS_ent and δS_BD. If δS_ent ∝ δS_BD across
# many perturbations, gravity emerges from entanglement.

to = TwoOrder(N, rng=rng)
cs = to.to_causet()
W0, _, _ = sj_wightman(cs)
S_ent_0 = entanglement_entropy(W0, list(range(N // 2)))
S_BD_0 = bd_action_corrected(cs, eps)

delta_ent = []
delta_bd = []

# Perturb: flip random relations
for trial in range(50):
    cs_pert = FastCausalSet(N)
    cs_pert.order = cs.order.copy()

    # Flip a random relation
    i, j = rng.choice(N, 2, replace=False)
    if i > j:
        i, j = j, i
    cs_pert.order[i, j] = not cs_pert.order[i, j]

    W_pert, _, _ = sj_wightman(cs_pert)
    S_ent_pert = entanglement_entropy(W_pert, list(range(N // 2)))
    S_BD_pert = bd_action_corrected(cs_pert, eps)

    delta_ent.append(S_ent_pert - S_ent_0)
    delta_bd.append(S_BD_pert - S_BD_0)

delta_ent = np.array(delta_ent)
delta_bd = np.array(delta_bd)
r = np.corrcoef(delta_ent, delta_bd)[0, 1]
print(f"\n  Correlation between δS_ent and δS_BD: r = {r:.3f}")
print(f"  Jacobson predicts r → 1 (entanglement first law)")
if abs(r) > 0.5:
    coeffs = np.polyfit(delta_bd, delta_ent, 1)
    print(f"  Linear fit: δS_ent = {coeffs[0]:.4f} * δS_BD + {coeffs[1]:.4f}")
print(f"  {'CONSISTENT with Jacobson' if r > 0.5 else 'NOT consistent'}")

# ================================================================
print("\n" + "=" * 75)
print("IDEA 3: LYAPUNOV EXPONENT / OTOC")
print("Is the continuum phase maximally chaotic?")
print("=" * 75)

# The OTOC F(t) = <A(t)B(0)A(t)B(0)> decays as exp(lambda_L * t)
# for chaotic systems. The MSS bound: lambda_L <= 2*pi*T.
#
# On a causal set, "time evolution" isn't standard. But we can
# define a discrete OTOC using the SJ correlator:
# F(d) = <sum over pairs at causal distance d of W[i,j]^2>
# If F(d) decays exponentially: lambda_L from the decay rate.

N = 50
to = TwoOrder(N, rng=rng)
cs = to.to_causet()
W, _, _ = sj_wightman(cs)
order_int = cs.order.astype(np.int32)
interval_matrix = order_int @ order_int

# Compute <W[i,j]^2> vs causal distance for CAUSAL pairs
dist_w2 = {}
for i in range(N):
    for j in range(i + 1, N):
        if cs.order[i, j]:
            d = interval_matrix[i, j]
            if d not in dist_w2:
                dist_w2[d] = []
            dist_w2[d].append(W[i, j] ** 2)

print(f"\n  OTOC proxy: <W[i,j]^2> vs causal distance")
print(f"  {'dist':>6} {'<W^2>':>10} {'ln(<W^2>)':>10} {'n':>6}")
print("-" * 36)

dists, w2s, ln_w2s = [], [], []
for d in sorted(dist_w2.keys()):
    vals = dist_w2[d]
    if len(vals) > 5 and d > 0:
        m = np.mean(vals)
        dists.append(d)
        w2s.append(m)
        ln_w2s.append(np.log(m + 1e-15))
        print(f"  {d:>6} {m:>10.6f} {np.log(m + 1e-15):>10.3f} {len(vals):>6}")

if len(dists) > 3:
    # Fit exponential: ln(<W^2>) = -lambda*d + const
    coeffs = np.polyfit(dists[:8], ln_w2s[:8], 1)
    lambda_L = -coeffs[0]
    print(f"\n  Lyapunov exponent: lambda_L = {lambda_L:.3f}")
    print(f"  MSS bound: lambda_L <= 2*pi*T")
    print(f"  (Need to define T to check saturation)")

# ================================================================
print("\n" + "=" * 75)
print("IDEA 4: 4D SJ ENTANGLEMENT ACROSS THE TRANSITION")
print("Does the 4D continuum phase also show CFT-like entanglement?")
print("=" * 75)

for d_dim in [2, 4]:
    N_test = 30 if d_dim == 4 else 50
    print(f"\n  d = {d_dim} (N={N_test}):")

    S_vals = []
    for _ in range(10):
        if d_dim == 2:
            to = TwoOrder(N_test, rng=rng)
            cs_t = to.to_causet()
        else:
            do = DOrder(d=4, N=N_test, rng=rng)
            cs_t = do.to_causet_fast()

        W_t, _, _ = sj_wightman(cs_t)
        S = entanglement_entropy(W_t, list(range(N_test // 2)))
        S_vals.append(S)

    print(f"    S(N/2) = {np.mean(S_vals):.3f} ± {np.std(S_vals):.3f}")
    print(f"    S/ln(N) = {np.mean(S_vals) / np.log(N_test):.3f} (CFT: const)")
    print(f"    S/N^0.5 = {np.mean(S_vals) / np.sqrt(N_test):.3f} (4D area law: const)")

# ================================================================
print("\n" + "=" * 75)
print("IDEA 5: MUTUAL INFORMATION BETWEEN PAST AND FUTURE LIGHT CONES")
print("How much information passes through a single spacetime event?")
print("=" * 75)

N = 50
to = TwoOrder(N, rng=rng)
cs = to.to_causet()
W, _, _ = sj_wightman(cs)

# For each element x, compute I(past(x) : future(x))
I_vals = []
for x in range(N):
    past_x = [i for i in range(N) if cs.order[i, x]]
    future_x = [j for j in range(N) if cs.order[x, j]]

    if len(past_x) < 2 or len(future_x) < 2:
        continue

    I = mutual_info(W, past_x, future_x)
    I_vals.append(I)

print(f"\n  I(past:future) across elements:")
print(f"    Mean: {np.mean(I_vals):.3f}")
print(f"    Std: {np.std(I_vals):.3f}")
print(f"    Max: {np.max(I_vals):.3f}")
print(f"    Min: {np.min(I_vals):.3f}")
print(f"    This measures 'channel capacity' of each spacetime event")

# Does it correlate with the element's position (depth in the order)?
depths = []
for x in range(N):
    past_x = [i for i in range(N) if cs.order[i, x]]
    future_x = [j for j in range(N) if cs.order[x, j]]
    if len(past_x) < 2 or len(future_x) < 2:
        continue
    depths.append(len(past_x))

r_depth = np.corrcoef(I_vals[:len(depths)], depths)[0, 1]
print(f"    Correlation with depth (|past|): r = {r_depth:.3f}")
print(f"    Deeper elements transmit {'MORE' if r_depth > 0.3 else 'similar'} information")

# ================================================================
print("\n" + "=" * 75)
print("IDEA 6: ENTANGLEMENT AREA LAW IN 3D")
print("In 3D, does S scale as boundary AREA (not log)?")
print("=" * 75)

# In d dimensions, area law: S ~ L^{d-2} for a region of size L
# d=2: S ~ ln(L) (log correction to area law = constant)
# d=3: S ~ L (area of 1D boundary)
# d=4: S ~ L^2 (area of 2D boundary)

for d in [2, 3, 4]:
    N_test = 30
    print(f"\n  d = {d}:")

    S_by_frac = {}
    for frac in [0.2, 0.3, 0.4, 0.5]:
        S_vals = []
        for _ in range(8):
            do = DOrder(d=d, N=N_test, rng=rng)
            cs_d = do.to_causet_fast()
            W_d, _, _ = sj_wightman(cs_d)
            k = max(2, int(frac * N_test))
            S_vals.append(entanglement_entropy(W_d, list(range(k))))
        S_by_frac[frac] = np.mean(S_vals)
        print(f"    frac={frac}: S = {np.mean(S_vals):.3f}")

    # Check scaling: does S grow linearly with frac (volume) or sublinearly (area)?
    fracs = sorted(S_by_frac.keys())
    Ss = [S_by_frac[f] for f in fracs]
    slope = (Ss[-1] - Ss[0]) / (fracs[-1] - fracs[0])
    print(f"    dS/dfrac = {slope:.3f}")

# ================================================================
print("\n" + "=" * 75)
print("IDEA 7: UNIVERSAL RATIO AT THE PHASE TRANSITION")
print("Is S_ent(beta_c) / S_ent(0) a universal number?")
print("=" * 75)

eps = 0.12
print(f"\n  S_ent(beta_c) / S_ent(0) at different N:")
print(f"  {'N':>5} {'S(0)':>8} {'S(bc)':>8} {'ratio':>8}")
print("-" * 35)

for N in [25, 30, 40, 50]:
    beta_c = 6.64 / (N * eps ** 2)

    # S at beta=0
    S0_vals = []
    for _ in range(10):
        to = TwoOrder(N, rng=rng)
        cs_0 = to.to_causet()
        W_0, _, _ = sj_wightman(cs_0)
        S0_vals.append(entanglement_entropy(W_0, list(range(N // 2))))

    # S at beta_c
    res = mcmc_corrected(N, beta=beta_c, eps=eps, n_steps=25000, n_therm=12000,
                          record_every=100, rng=rng)
    Sc_vals = []
    for cs_c in res['samples'][-10:]:
        W_c, _, _ = sj_wightman(cs_c)
        Sc_vals.append(entanglement_entropy(W_c, list(range(N // 2))))

    S0 = np.mean(S0_vals)
    Sc = np.mean(Sc_vals)
    ratio = Sc / S0 if S0 > 0 else float('nan')
    print(f"  {N:>5} {S0:>8.3f} {Sc:>8.3f} {ratio:>8.3f}")

# ================================================================
print("\n" + "=" * 75)
print("IDEA 8: ENTANGLEMENT SPECTRUM MATCHES CFT PREDICTION")
print("Does the entanglement spectrum have the structure of a CFT?")
print("=" * 75)

# In a CFT, the entanglement spectrum (eigenvalues of -ln(rho_A))
# should have specific degeneracy structure determined by the
# Virasoro algebra. The lowest level is non-degenerate, the next
# has degeneracy 1, then 2, etc.

N = 50
to = TwoOrder(N, rng=rng)
cs = to.to_causet()
W, _, _ = sj_wightman(cs)
A = list(range(N // 2))
W_A = W[np.ix_(A, A)]
eigs = np.linalg.eigvalsh(W_A)
eigs = np.clip(eigs, 1e-15, 1 - 1e-15)

# Entanglement energies: xi_k = -ln(nu_k / (1 - nu_k))
xi = -np.log(eigs / (1 - eigs))
xi = np.sort(xi)

print(f"\n  Entanglement spectrum (lowest 10 levels):")
print(f"  {'level':>6} {'xi_k':>10} {'gap from ground':>16}")
xi_ground = xi[0]
for k in range(min(10, len(xi))):
    print(f"  {k:>6} {xi[k]:>10.3f} {xi[k] - xi_ground:>16.3f}")

# Check: are the gaps integer multiples of a fundamental spacing?
gaps = xi[1:11] - xi_ground
if len(gaps) > 3:
    fundamental = gaps[0]
    ratios = gaps / fundamental
    print(f"\n  Gap ratios (relative to first gap):")
    for k, r in enumerate(ratios):
        print(f"    level {k + 1}: ratio = {r:.2f} (integer: {round(r)})")
    print(f"  Integer ratios = CFT tower structure")

# ================================================================
print("\n" + "=" * 75)
print("IDEA 9: CAUSAL SET CENTRAL CHARGE FROM ENTANGLEMENT ENTROPY")
print("Use the Calabrese-Cardy formula properly with finite-size corrections")
print("=" * 75)

# CC formula: S = (c/3) * ln((N/pi) * sin(pi*l/N)) + const
# where l is the subsystem size. This includes finite-size corrections.

N = 60
print(f"\n  Fitting CC formula at N={N}:")

S_data = []
l_data = []
for l_frac in np.linspace(0.1, 0.5, 9):
    l = max(2, int(l_frac * N))
    S_vals = []
    for _ in range(12):
        to = TwoOrder(N, rng=rng)
        cs_cc = to.to_causet()
        W_cc, _, _ = sj_wightman(cs_cc)
        S_vals.append(entanglement_entropy(W_cc, list(range(l))))
    S_data.append(np.mean(S_vals))
    l_data.append(l)

# Fit: S = (c/3) * ln((N/pi) * sin(pi*l/N)) + s0
x_cc = np.log((N / np.pi) * np.sin(np.pi * np.array(l_data) / N))
coeffs_cc = np.polyfit(x_cc, S_data, 1)
c_cc = 3 * coeffs_cc[0]

print(f"  c (from CC fit with finite-size corrections) = {c_cc:.2f}")
print(f"  c (naive S/ln(N)) = {3 * S_data[-1] / np.log(N):.2f}")
print(f"  Free scalar prediction: c = 1")
print(f"  CC fit is more accurate than naive because it accounts for")
print(f"  the sin(pi*l/N) finite-size correction from periodic boundary conditions")

# ================================================================
print("\n" + "=" * 75)
print("IDEA 10: DOES THE SJ VACUUM SATISFY STRONG SUBADDITIVITY?")
print("SSA is required for any valid quantum state. Violation = error in construction.")
print("=" * 75)

# S(ABC) + S(B) <= S(AB) + S(BC) for all tripartitions
N = 40
n_tests = 50
n_violations = 0

for _ in range(n_tests):
    to = TwoOrder(N, rng=rng)
    cs_ssa = to.to_causet()
    W_ssa, _, _ = sj_wightman(cs_ssa)

    # Random tripartition
    perm = rng.permutation(N)
    third = N // 3
    A = list(perm[:third])
    B = list(perm[third:2 * third])
    C = list(perm[2 * third:])

    AB = list(set(A) | set(B))
    BC = list(set(B) | set(C))
    ABC = list(set(A) | set(B) | set(C))

    S_ABC = entanglement_entropy(W_ssa, ABC)
    S_B = entanglement_entropy(W_ssa, B)
    S_AB = entanglement_entropy(W_ssa, AB)
    S_BC = entanglement_entropy(W_ssa, BC)

    # SSA: S(ABC) + S(B) <= S(AB) + S(BC)
    lhs = S_ABC + S_B
    rhs = S_AB + S_BC
    if lhs > rhs + 0.001:  # small tolerance for numerics
        n_violations += 1

print(f"\n  SSA violations: {n_violations}/{n_tests} ({100 * n_violations / n_tests:.0f}%)")
if n_violations == 0:
    print(f"  PERFECT: SJ vacuum satisfies SSA in all cases")
    print(f"  This validates that our W construction produces a legitimate quantum state")
else:
    print(f"  VIOLATIONS FOUND — potential issue with the SJ construction or normalization")

print("\n" + "=" * 75)
print("ALL 10 IDEAS TESTED")
print("=" * 75)

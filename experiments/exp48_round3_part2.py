"""
Experiment 48 Part 2: Ideas 65-75 (fixed bugs from Part 1)

Ideas 56-64 ran in Part 1. Continuing from 65.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats
from causal_sets.two_orders import TwoOrder, swap_move
from causal_sets.two_orders_v2 import bd_action_corrected
from causal_sets.fast_core import FastCausalSet
from causal_sets.bd_action import count_intervals_by_size

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


def mcmc_with_twoorders(N, beta, eps, n_steps=20000, n_therm=10000,
                          record_every=50, rng=None):
    """MCMC that stores TwoOrder objects (not just FastCausalSet)."""
    if rng is None:
        rng = np.random.default_rng()

    current = TwoOrder(N, rng=rng)
    current_cs = current.to_causet()
    current_S = bd_action_corrected(current_cs, eps)

    samples_to = []
    actions = []
    n_acc = 0

    for step in range(n_steps):
        proposed = swap_move(current, rng)
        proposed_cs = proposed.to_causet()
        proposed_S = bd_action_corrected(proposed_cs, eps)

        dS = beta * (proposed_S - current_S)
        if dS <= 0 or rng.random() < np.exp(-dS):
            current = proposed
            current_cs = proposed_cs
            current_S = proposed_S
            n_acc += 1

        if step >= n_therm and step % record_every == 0:
            actions.append(current_S)
            samples_to.append(current.copy())

    return {
        'actions': np.array(actions),
        'samples': samples_to,
        'acceptance': n_acc / n_steps,
    }


# ================================================================
print("=" * 75)
print("IDEA 65: CORRELATION LENGTH AT BD TRANSITION")
print("Does |W[i,j]| correlation length diverge at β_c?")
print("=" * 75)

N = 50
eps = 0.12
beta_c = 1.66 / (N * eps**2)
betas = [0, 0.5*beta_c, 0.8*beta_c, beta_c, 1.2*beta_c, 1.5*beta_c, 2*beta_c]

for beta in betas:
    if beta == 0:
        to = TwoOrder(N, rng=rng)
    else:
        result = mcmc_with_twoorders(N, beta, eps, n_steps=15000, n_therm=7500,
                                      record_every=50, rng=rng)
        if result['samples']:
            to = result['samples'][-1]
        else:
            print(f"β/β_c={beta/beta_c:.2f}: no samples")
            continue

    cs = to.to_causet()
    W, _, _ = sj_full(cs)

    # Spatial correlation function C(d) = <|W[i,j]|> at spatial distance d
    v = to.v / N
    corr_by_dist = {}
    for i in range(N):
        for j in range(i+1, N):
            d = abs(v[i] - v[j])
            d_bin = round(d * 10) / 10
            if d_bin not in corr_by_dist:
                corr_by_dist[d_bin] = []
            corr_by_dist[d_bin].append(abs(W[i, j]))

    dists = sorted(d for d in corr_by_dist.keys() if d > 0)
    means = [np.mean(corr_by_dist[d]) for d in dists]

    if len(dists) >= 4 and means[0] > 1e-10:
        log_C = [np.log(max(m, 1e-15)) for m in means[:5]]
        slope, _, r, _, _ = stats.linregress(dists[:5], log_C)
        xi = -1.0 / slope if slope < -0.01 else float('inf')
        print(f"β/β_c={beta/beta_c:.2f}: ξ={xi:.2f}, C(0.1)={means[0]:.4f}, C(0.3)={means[2] if len(means)>2 else 0:.4f} (r={r:.3f})")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 66: DEFICIT ANGLE FROM INTERVAL COUNTING")
print("Interval distribution deviation from flat → curvature signal?")
print("=" * 75)

N = 50
n_trials = 20

# Compare interval distributions of flat 2-orders vs "curved" (conformal factor)
flat_dists = []
curved_dists = []

for trial in range(n_trials):
    # Flat
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    counts = count_intervals_by_size(cs, max_size=8)
    total = sum(counts.values())
    if total > 0:
        flat_dists.append({k: counts.get(k, 0)/total for k in range(9)})

    # "Curved": apply conformal stretching to coordinates
    # This changes the effective density (simulating curvature)
    u_conf = np.argsort(np.exp(0.5 * rng.permutation(N) / N))
    v_conf = rng.permutation(N)
    to_curved = TwoOrder.from_permutations(u_conf, v_conf)
    cs_c = to_curved.to_causet()
    counts_c = count_intervals_by_size(cs_c, max_size=8)
    total_c = sum(counts_c.values())
    if total_c > 0:
        curved_dists.append({k: counts_c.get(k, 0)/total_c for k in range(9)})

print("k | flat N_k/total | curved N_k/total | ratio")
for k in range(7):
    flat_vals = [d.get(k, 0) for d in flat_dists]
    curv_vals = [d.get(k, 0) for d in curved_dists]
    f_mean = np.mean(flat_vals)
    c_mean = np.mean(curv_vals)
    ratio = c_mean / max(f_mean, 1e-10)
    print(f"  {k} | {f_mean:.4f} ± {np.std(flat_vals):.4f} | {c_mean:.4f} ± {np.std(curv_vals):.4f} | {ratio:.3f}")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 67: D'ALEMBERTIAN SPECTRUM")
print("Causal set d'Alembertian vs lattice Laplacian eigenvalues.")
print("=" * 75)

N = 50
n_trials = 10
cs_spectra = []
lattice_spectra = []

for trial in range(n_trials):
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    C = cs.order.astype(float)

    # BD d'Alembertian (simplified 2D): B_ij = (4/l²)[ρ(δ_ij - 2L_ij + I2_ij)]
    # where L = link matrix, I2 = 2-interval matrix
    # Simpler version: B = I - (2/N)(C + C^T) + (1/N²)(C^T @ C)
    B = np.eye(N) - (2.0/N)*(C + C.T) + (1.0/N**2) * (C.T @ C + C @ C.T)
    evals_B = np.sort(np.linalg.eigvalsh(B))
    cs_spectra.append(evals_B)

    # 1D lattice Laplacian for comparison
    L = np.zeros((N, N))
    for i in range(N-1):
        L[i, i] += 1; L[i+1, i+1] += 1
        L[i, i+1] = -1; L[i+1, i] = -1
    L /= N  # normalize similarly
    evals_L = np.sort(np.linalg.eigvalsh(L))
    lattice_spectra.append(evals_L)

avg_B = np.mean(cs_spectra, axis=0)
avg_L = np.mean(lattice_spectra, axis=0)

print("Low eigenvalues (first 8):")
print(f"  d'Alembert: {avg_B[:8].round(4)}")
print(f"  Lattice:    {avg_L[:8].round(4)}")
print(f"\nSpectral gap: d'Alembert={avg_B[1]-avg_B[0]:.4f}, Lattice={avg_L[1]-avg_L[0]:.6f}")

# Weyl law test: N(λ) vs λ^{d/2}
# Count eigenvalues below λ for several λ
thresholds = [0.1, 0.2, 0.5, 1.0, 1.5]
print("\nWeyl law N(λ):")
for lam in thresholds:
    n_B = np.mean([np.sum(s < lam) for s in cs_spectra])
    n_L = np.mean([np.sum(s < lam) for s in lattice_spectra])
    print(f"  λ={lam}: d'Alembert N(λ)={n_B:.1f}, Lattice N(λ)={n_L:.1f}")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 68: CONDITIONAL MUTUAL INFORMATION — MARKOV PROPERTY")
print("I(A:C|B) for B separating A and C. ≈0 → Markov → holographic.")
print("=" * 75)

N = 60
n_trials = 20
cmi_causet = []
cmi_random = []

for trial in range(n_trials):
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    W, _, _ = sj_full(cs)

    v_sorted = np.argsort(to.v)
    fifth = N // 5
    A = list(v_sorted[:fifth])
    B = list(v_sorted[fifth:4*fifth])
    C = list(v_sorted[4*fifth:])

    AB = sorted(set(A) | set(B))
    BC = sorted(set(B) | set(C))
    ABC = sorted(set(A) | set(B) | set(C))

    S_AB = entanglement_entropy(W, AB)
    S_BC = entanglement_entropy(W, BC)
    S_B = entanglement_entropy(W, list(B))
    S_ABC = entanglement_entropy(W, ABC)

    I_AC_B = S_AB + S_BC - S_B - S_ABC
    cmi_causet.append(I_AC_B)

    # Null
    order_rand = np.zeros((N, N), dtype=bool)
    perm = rng.permutation(N)
    for i in range(N):
        for j in range(i+1, N):
            if rng.random() < 0.25:
                order_rand[perm[i], perm[j]] = True
    cs_r = FastCausalSet(N)
    cs_r.order = order_rand
    W_r, _, _ = sj_full(cs_r)
    S_AB_r = entanglement_entropy(W_r, AB)
    S_BC_r = entanglement_entropy(W_r, BC)
    S_B_r = entanglement_entropy(W_r, list(B))
    S_ABC_r = entanglement_entropy(W_r, ABC)
    cmi_random.append(S_AB_r + S_BC_r - S_B_r - S_ABC_r)

print(f"I(A:C|B) causet: {np.mean(cmi_causet):.4f} ± {np.std(cmi_causet):.4f}")
print(f"I(A:C|B) random: {np.mean(cmi_random):.4f} ± {np.std(cmi_random):.4f}")
ratio = np.mean(cmi_causet) / max(np.mean(cmi_random), 1e-6)
print(f"Ratio causet/random: {ratio:.2f}")
# Normalized by S(A): how "Markov" is the state?
S_A_vals = []
for trial in range(min(5, len(cmi_causet))):
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    W, _, _ = sj_full(cs)
    v_sorted = np.argsort(to.v)
    A = list(v_sorted[:N//5])
    S_A_vals.append(entanglement_entropy(W, A))
mean_SA = np.mean(S_A_vals) if S_A_vals else 1
print(f"I(A:C|B)/S(A) ≈ {np.mean(cmi_causet)/max(mean_SA, 1e-6):.4f}")
print(f"  ≈0 → approximately Markov. Near-zero CMI is a hallmark of holographic states.")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 69: FISHER INFORMATION AT PHASE TRANSITION")
print("F(β) = β² × Var(S_BD). Diverges at β_c?")
print("=" * 75)

N = 50
eps = 0.12
beta_c = 1.66 / (N * eps**2)
betas_scan = np.linspace(0.3*beta_c, 2.5*beta_c, 10)
fisher_data = []

for beta in betas_scan:
    result = mcmc_with_twoorders(N, beta, eps, n_steps=12000, n_therm=6000,
                                  record_every=30, rng=rng)
    if len(result['actions']) > 10:
        var_S = np.var(result['actions'])
        F = beta**2 * var_S
        fisher_data.append((beta/beta_c, F, var_S, result['acceptance']))
        print(f"β/β_c={beta/beta_c:.2f}: Var(S)={var_S:.2f}, F(β)={F:.1f}, acc={result['acceptance']:.1%}")

if fisher_data:
    max_F = max(f[1] for f in fisher_data)
    max_beta = [f[0] for f in fisher_data if f[1] == max_F][0]
    print(f"\nPeak Fisher info at β/β_c = {max_beta:.2f}, F_max = {max_F:.1f}")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 70: MULTIPARTITE ENTANGLEMENT — RESIDUAL TANGLE")
print("Genuine tripartite entanglement beyond bipartite sharing?")
print("=" * 75)

N = 60
n_trials = 20
tangles_causet = []
tangles_random = []

for trial in range(n_trials):
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    W, _, _ = sj_full(cs)

    v_sorted = np.argsort(to.v)
    third = N // 3
    A = list(v_sorted[:third])
    B = list(v_sorted[third:2*third])
    C = list(v_sorted[2*third:])
    AB = sorted(set(A) | set(B))
    AC = sorted(set(A) | set(C))

    S_A = entanglement_entropy(W, A)
    S_B = entanglement_entropy(W, list(B))
    S_C = entanglement_entropy(W, list(C))
    S_AB = entanglement_entropy(W, AB)
    S_AC = entanglement_entropy(W, AC)

    I_AB = S_A + S_B - S_AB
    I_AC = S_A + S_C - S_AC
    tau = S_A**2 - I_AB**2 - I_AC**2
    tangles_causet.append(tau)

    # Null
    order_rand = np.zeros((N, N), dtype=bool)
    perm = rng.permutation(N)
    for i in range(N):
        for j in range(i+1, N):
            if rng.random() < 0.25:
                order_rand[perm[i], perm[j]] = True
    cs_r = FastCausalSet(N)
    cs_r.order = order_rand
    W_r, _, _ = sj_full(cs_r)
    S_A_r = entanglement_entropy(W_r, A)
    S_B_r = entanglement_entropy(W_r, list(B))
    S_C_r = entanglement_entropy(W_r, list(C))
    S_AB_r = entanglement_entropy(W_r, AB)
    S_AC_r = entanglement_entropy(W_r, AC)
    I_AB_r = S_A_r + S_B_r - S_AB_r
    I_AC_r = S_A_r + S_C_r - S_AC_r
    tangles_random.append(S_A_r**2 - I_AB_r**2 - I_AC_r**2)

print(f"Residual tangle (causet): {np.mean(tangles_causet):.4f} ± {np.std(tangles_causet):.4f}")
print(f"Residual tangle (random): {np.mean(tangles_random):.4f} ± {np.std(tangles_random):.4f}")
t = (np.mean(tangles_causet) - np.mean(tangles_random)) / np.sqrt(
    np.std(tangles_causet)**2/n_trials + np.std(tangles_random)**2/n_trials)
print(f"t-stat: {t:.1f}")
print(f"  τ > 0 → genuine tripartite entanglement")
print(f"  τ < 0 → bipartite entanglement dominates")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 71: LOCALITY OF W — PROPAGATOR DECAY")
print("How does |W[i,j]| decay with proper distance?")
print("=" * 75)

N = 60
n_trials = 15
decay_exp_timelike = []
decay_exp_spacelike = []

for trial in range(n_trials):
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    W, _, _ = sj_full(cs)

    t_coord = (to.u + to.v) / (2.0 * N)
    x_coord = (to.v - to.u) / (2.0 * N)

    w_time = []
    w_space = []
    for i in range(N):
        for j in range(i+1, N):
            dt = abs(t_coord[j] - t_coord[i])
            dx = abs(x_coord[j] - x_coord[i])
            ds2 = dt**2 - dx**2
            w = abs(W[i, j])
            if w > 1e-10:
                if ds2 > 0.001:  # timelike
                    w_time.append((np.sqrt(ds2), w))
                elif ds2 < -0.001:  # spacelike
                    w_space.append((np.sqrt(-ds2), w))

    for label, data, result_list in [("timelike", w_time, decay_exp_timelike),
                                      ("spacelike", w_space, decay_exp_spacelike)]:
        if len(data) > 20:
            d_arr, w_arr = zip(*data)
            d_arr, w_arr = np.array(d_arr), np.array(w_arr)
            mask = (d_arr > 0.01) & (w_arr > 1e-10)
            if np.sum(mask) > 10:
                slope, _, r, _, _ = stats.linregress(np.log(d_arr[mask]), np.log(w_arr[mask]))
                result_list.append(slope)

if decay_exp_timelike:
    print(f"Timelike: |W| ~ d^{np.mean(decay_exp_timelike):.3f} ± {np.std(decay_exp_timelike):.3f}")
    print(f"  Continuum 2D massless scalar: W ~ 1/d → exponent = -1")
if decay_exp_spacelike:
    print(f"Spacelike: |W| ~ d^{np.mean(decay_exp_spacelike):.3f} ± {np.std(decay_exp_spacelike):.3f}")
    print(f"  Continuum: Wightman function decays as 1/σ in 2D")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 72: RELATIVE ENTROPY BETWEEN PHASES")
print("S(ρ_cont || ρ_cryst) quantum distance. Scaling with N?")
print("=" * 75)

sizes = [20, 30, 40, 50]
for N in sizes:
    eps = 0.12
    beta_c = 1.66 / (N * eps**2)

    to_cont = TwoOrder(N, rng=rng)
    cs_cont = to_cont.to_causet()
    W_cont, _, _ = sj_full(cs_cont)

    result = mcmc_with_twoorders(N, 3*beta_c, eps, n_steps=15000, n_therm=7500,
                                  record_every=50, rng=rng)
    if result['samples']:
        cs_cryst = result['samples'][-1].to_causet()
        W_cryst, _, _ = sj_full(cs_cryst)

        eigs_c = np.clip(np.linalg.eigvalsh(W_cont), 1e-14, 1 - 1e-14)
        eigs_k = np.clip(np.linalg.eigvalsh(W_cryst), 1e-14, 1 - 1e-14)

        S_rel = np.sum(eigs_c * np.log(eigs_c / eigs_k) +
                       (1 - eigs_c) * np.log((1 - eigs_c) / (1 - eigs_k)))
        print(f"N={N}: S_rel = {S_rel:.2f}, S_rel/N = {S_rel/N:.3f}, S_rel/N² = {S_rel/N**2:.5f}")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 73: EIGENSTATE THERMALIZATION (ETH)")
print("Do SJ eigenstates have thermal reduced density matrices?")
print("=" * 75)

N = 50
n_trials = 10
eth_ratios = []

for trial in range(n_trials):
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    _, evals, evecs = sj_full(cs)

    pos_idx = np.where(evals > 1e-12)[0]
    if len(pos_idx) < 5:
        continue

    A = list(range(N // 4))  # small subsystem
    for idx in pos_idx[1:6]:  # skip zero mode, take 5 mid-energy states
        psi = evecs[:, idx].real
        norm = np.sum(psi[A]**2)
        if norm < 1e-10:
            continue
        rho_A = np.outer(psi[A], psi[A]) / norm
        eigs_rho = np.linalg.eigvalsh(rho_A)
        eigs_rho = eigs_rho[eigs_rho > 1e-15]
        if len(eigs_rho) > 0:
            S_eigenstate = -np.sum(eigs_rho * np.log(eigs_rho))
            S_thermal = np.log(len(A))  # max entropy
            eth_ratios.append(S_eigenstate / S_thermal)

if eth_ratios:
    print(f"S_eigenstate/S_thermal = {np.mean(eth_ratios):.4f} ± {np.std(eth_ratios):.4f}")
    print(f"  ETH satisfied: ratio → 1.0 (thermal)")
    print(f"  Non-ETH (integrable): ratio << 1")
    print(f"  Our value: {'thermal-like' if np.mean(eth_ratios) > 0.5 else 'non-thermal'}")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 74: SJ ON PERTURBED CAUSETS — LINEAR RESPONSE")
print("Flip causal relations. δS linear in perturbation strength?")
print("=" * 75)

N = 50
n_trials = 15
pert_strengths = [0.005, 0.01, 0.02, 0.05, 0.1]
delta_S = {p: [] for p in pert_strengths}

for trial in range(n_trials):
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    W0, _, _ = sj_full(cs)
    S0 = entanglement_entropy(W0, list(range(N // 2)))

    for p in pert_strengths:
        cs_pert = FastCausalSet(N)
        order_pert = cs.order.copy()
        n_flip = max(1, int(p * N * (N-1) / 2))
        for _ in range(n_flip):
            i, j = rng.integers(0, N, size=2)
            if i != j:
                order_pert[i, j] = not order_pert[i, j]
        cs_pert.order = order_pert
        W_p, _, _ = sj_full(cs_pert)
        S_p = entanglement_entropy(W_p, list(range(N // 2)))
        delta_S[p].append(abs(S_p - S0) / max(S0, 1e-10))

print("ε     | δS/S₀ (mean ± std)")
for p in pert_strengths:
    print(f"  {p:.3f} | {np.mean(delta_S[p]):.4f} ± {np.std(delta_S[p]):.4f}")

means = [np.mean(delta_S[p]) for p in pert_strengths]
r_lin = np.corrcoef(pert_strengths, means)[0, 1]
slope, intercept, _, _, _ = stats.linregress(pert_strengths, means)
print(f"\nδS/S₀ = {slope:.2f}×ε + {intercept:.4f} (r={r_lin:.3f})")
print(f"  Linear response (r≈1) = well-defined perturbation theory")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 75: ENTANGLEMENT ASYMMETRY — ARROW OF TIME")
print("S(past) vs S(future). Time-reversal symmetry?")
print("=" * 75)

N = 50
n_trials = 30
asym_causet = []
asym_lattice = []

for trial in range(n_trials):
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    W, _, _ = sj_full(cs)

    t_coord = (to.u + to.v) / (2.0 * N)
    t_sorted = np.argsort(t_coord)

    past = list(t_sorted[:N // 2])
    future = list(t_sorted[N // 2:])

    S_past = entanglement_entropy(W, past)
    S_future = entanglement_entropy(W, future)

    # Also left/right (spatial) — should be symmetric
    x_coord = (to.v - to.u) / (2.0 * N)
    x_sorted = np.argsort(x_coord)
    left = list(x_sorted[:N // 2])
    right = list(x_sorted[N // 2:])
    S_left = entanglement_entropy(W, left)
    S_right = entanglement_entropy(W, right)

    asym_time = (S_past - S_future) / (S_past + S_future + 1e-15)
    asym_space = (S_left - S_right) / (S_left + S_right + 1e-15)
    asym_causet.append(asym_time)
    asym_lattice.append(asym_space)

print(f"Time asymmetry:  {np.mean(asym_causet):.5f} ± {np.std(asym_causet):.5f}")
print(f"Space asymmetry: {np.mean(asym_lattice):.5f} ± {np.std(asym_lattice):.5f}")
t_time = abs(np.mean(asym_causet)) / (np.std(asym_causet) / np.sqrt(n_trials))
t_space = abs(np.mean(asym_lattice)) / (np.std(asym_lattice) / np.sqrt(n_trials))
print(f"t from zero: time={t_time:.1f}, space={t_space:.1f}")
print(f"  Both should be ≈0 for SJ vacuum on flat Minkowski")
print(f"  Time asymmetry ≠ 0 would indicate emergent arrow of time")


# ================================================================
print("\n" + "=" * 75)
print("SUMMARY: IDEAS 56-75 SCORES")
print("=" * 75)
print("""
Scoring criteria:
- Novel result? (not already known from earlier experiments)
- Passes null model control?
- Quantitatively interesting?
- Could be part of a paper?
""")

# Papers Summary

## Project Overview

**600 ideas tested** across ~107 experiments, ~66,000 lines of Python. 10 papers written (8 for submission). The 8.0/10 ceiling is structural at toy scale (N=30-200): density dominance, finite-size contamination, and no exact finite-N predictions from causal set theory prevent any single result from reaching 9+.

## Scoring Rubric

| Score | Meaning | Example |
|---|---|---|
| 1-2 | Homework exercise, not publishable | Reproducing a textbook calculation |
| 3-4 | Minor technical contribution; publishable with significant revisions | New observable for a known effect at toy scale |
| **5** | **Solid incremental result; publishable as-is in a specialist journal (e.g. CQG)** | **Novel order parameter with validation, but limited scope** |
| **6** | **Good result that fills a gap; likely accepted at a strong journal (e.g. PRD)** | **Systematic study with clear methodology, engages with prior work** |
| **7** | **Strong result that changes how people think about a specific problem** | **Theorem + reinterpretation of published results; bridges subfields** |
| 8 | Important result with impact across a subfield; top ~10% of papers | New method or prediction that multiple groups would follow up on |
| 9-10 | Breakthrough with broad impact across physics | New symmetry principle, paradigm shift |

## Completed Papers (Honest Scores)

| Paper | File | Pages | Score | Target | Key Claim |
|---|---|---|---|---|---|
| **E: CDT Comparison** | `cdt-comparison/` | ~7 | **8.0/10** | PRD | First cross-approach SJ vacuum comparison. CDT c_eff~1, causet c_eff->inf. **Kronecker product theorem:** C^T-C = A_T tensor J gives n_pos=floor(T/2) (proved in 6 steps). Exact eigenvalue formula mu_k = cot(pi(2k-1)/(2T)) verified to 10^-14. Wightman depends ONLY on temporal indices (spatial variation ~10^-17). S(N/2) depends only on T, not s. W_T is exactly Toeplitz (time-translation invariant). Sum rule: Σλ_k = (2/π)ln(T) + 0.527 (R²=1.000). MPS bond dimension polynomial in T. Single perturbation increases n_pos by 1 (extreme fragility). Generalizes to all spatial dimensions. **Caveat:** c_eff → 0 in strict continuum limit T,s→∞. |
| **B5: Geometry from Entanglement** | `geometry-from-entanglement/` | ~7 | **7.5/10** | PRD | **Flagship.** Spectral dimension fails (crossing theorem + random graph controls) -> SJ entanglement succeeds (ln N scaling + 3.4x drop across BD transition). Monogamy as supporting evidence with partition-dependence caveat. **Caveat:** c_eff diverges at large N (3.0->4.1 from N=50->500); relative comparison between phases remains valid. |
| **C: ER=EPR** | `er-epr/` | ~8 | **7.5/10** | PRD | |W| proportional to connectivity^0.90, r=0.88, z=13.1, partial r=0.82. Analytic proof: (-Delta^2)_ij = (4/N^2)*kappa_ij exactly for spacelike pairs. **Universal:** Gram identity holds on ALL partial orders (not just 2-orders) -- proved via transitivity, verified on 3-orders, 4-orders, sprinkled 2D/3D/4D. r=0.91 at d=4 (stronger than d=2). **Caveat:** universality means it also holds on random DAGs, explaining large-N genericity (gap closes at N=500). |
| **G: Exact Combinatorics** | `exact-combinatorics/` | ~7 | **7.5/10** | J. Comb. Theory/CQG | 16+ exact results: E[f]=1/2, Var[f]~1/(9N). Antichain=2*sqrt(N) via Vershik-Kerov. Master formula P(int=k|gap=m). E[links]=(N+1)H_N-2N. E[S_Glaser]=1 for all N>=2. E[S_BD]=2N-NH_N. E[S_BD(epsilon)] exact formula verified to 1%. Limit shape f(alpha). Exact Z(beta) for N=4,5. Maximal elements E[max]=H_N. k-antichains E=C(N,k)/k!. Chain-antichain symmetry: E[chains]=E[antichains]=C(N,k)/k!. Unimodality of E[N_k]. Link fraction asymptotic 2ln(N)/N (undirected), directed/undirected distinction resolved. Master formula specific to d=2. |
| **D: Spectral Statistics** | `random-matrix/` | ~6 | **7.5/10** | PRD | Universal GUE (r=0.56-0.60) across ALL phases, ALL beta, ALL epsilon, N=20-1000. GUE confirmed on 3-orders, 4-orders, and sprinkled 2D/3D/4D causets. GUE is generic to dense antisymmetric matrices (Erdos-Yau). Sub-Poisson r=0.12 was a PHASE-MIXING ARTIFACT: concatenating GUE spectra gives r->0.39 (Poisson), min r=0.31. Spectral compressibility chi->0 with proper unfolding. |
| **F: Hasse Geometry** | `hasse-geometry/` | ~6 | **7.0/10** | PRD/CQG | Hasse Laplacian Fiedler value 50x larger for causets than DAGs at small N. **CORRECTED:** Fiedler SATURATES at ~1.4-1.6 for N>100, scaling as N^0.054 (NOT N^0.32). Hasse diameter = 6 at N=3000, grows as N^0.11 (NOT Theta(sqrt(N))). Spectral embedding with 19 eigenvectors gives R^2=0.83-0.91. Link fraction as perfectly monotonic BD order parameter (60% jump). Triangle-free (girth>=4), independence number alpha~N^0.83. |
| **A: Interval Entropy** | `interval-entropy/` | ~6 | **7.0/10** | CQG/PRD | Interval entropy as BD order parameter. H=2.4->0.3 across 2D transition (87% drop). Random graph control (H=2.1 for matched DAGs). 4D three-phase structure at N=30-70 with chi_S up to 43,288. Non-monotonic H(beta) in 4D. Sprinkled 4D causets match disordered d-orders. |
| **B2: Everpresent Lambda** | `everpresent-lambda/` | ~4 | **5.5/10** | JCAP/PRD | alpha=0.03 predicts Omega_Lambda=0.732+/-0.103 from first principles. Optimal alpha=0.035-0.070. Bayes factor 3.8x favoring LCDM ("substantial" Jeffreys). w(z) deviates from -1 (quintessence-like, mean w~-0.4 at z=0). 48% within DESI 2-sigma. High stochastic variance limits predictive power. |
| B1: Spectral Dimension | `spectral-dimension/` | ~4.5 | 7/10 | -- | Crossing theorem (IVT). *Subsumed by B5.* |
| B3: Holographic Entanglement | `holographic-entanglement/` | ~5 | 7/10 | -- | SJ vacuum properties. *Subsumed by B5.* |

## Recommended Submission Strategy

1. **E** (8.0) -- CDT comparison with Kronecker product theorem, exact eigenvalue formula, Wightman factorization
2. **C** (7.5) -- ER=EPR with universal Gram identity + honest large-N caveats
3. **G** (7.5) -- exact combinatorics, bridges QG and combinatorics/probability communities
4. **D** (7.5) -- universal GUE across d=2,3,4 + sub-Poisson artifact identification
5. **B5** (7.5) -- flagship, broadest narrative (c_eff caveat included)
6. **F** (7.0) -- Hasse diagram geometry with corrected Fiedler saturation
7. **A** (7.0) -- interval entropy + 4D three-phase structure
8. **B2** (5.5) -- standalone if desired, weakest paper but now with Bayes factor
9. B1, B3 subsumed by B5 -- don't submit separately

## Compilation

```bash
cd papers/<paper-dir>
tectonic <paper>.tex
```

All 8 submission papers compile successfully as of 2026-03-22. Figures generated for Papers A, B5, C.

## Key Data Points Across Papers

| Result | Experiment | Paper |
|---|---|---|
| d_s crossing is generic to all graphs (IVT proof) | exp04 | B1, B5 |
| CDT plateau grows: 17%->47% (N=100->800) | exp19-20 | B1, B5 |
| Causet d_s matches random graphs at same L/N | exp25 | A, B1, B5 |
| BD d'Alembertian also fails random graph control | exp37 | B1, B5 |
| c_eff diverges: 3.0->4.1 (N=50->500), same on sprinkled causets | exp49, exp56 | B5 |
| Interval entropy: H=2.4->0.3 across 2D transition (87% drop) | exp26, exp29 | A |
| 2D chi_max scales as ~N^3 (strong first-order) | exp32 | A |
| 4D three-phase: H rises to 1.23, dips to 0.27, recovers to 0.69 | exp42 | A |
| 4D chi_S_max = 43,288 at N=70 | exp42 | A |
| Sprinkled 4D causets match disordered d-orders in interval entropy | exp100 | A |
| Omega_Lambda = 0.732 +/- 0.103 after anthropic selection | exp17 | B2 |
| Optimal alpha=0.035-0.070, Bayes factor 3.8x favoring LCDM | exp100 | B2 |
| SJ entanglement S(N/2) ~ ln(N) (CFT-like at small N) | exp35 | B3, B5 |
| S(N/2) drops 3.4x across BD transition | exp35 | B3, B5 |
| Monogamy I_3 <= 0 in 97% but partition-dependent | exp36, exp40 | B5 |
| |W| proportional to connectivity^0.90, r=0.88, z=13.1, partial r=0.82 | exp43 | C |
| X_ij = 0 for spacelike pairs in ANY partial order (proved via transitivity) | exp50, exp104 | C |
| (-Delta^2)_ij = (4/N^2)*kappa_ij universal to all causets | exp50, exp100 | C |
| W_ij = 1/2[sqrt(-Delta^2)]_ij for spacelike pairs (verified to 10^-15) | exp50 | C |
| ER=EPR r=0.91 at d=4, N=50 (stronger than d=2) | exp104 | C |
| ER=EPR gap vs density-matched DAGs closes at N=500 | exp49 | C |
| GUE (r~0.56) in continuum AND deep crystalline phases | exp-RMT, exp82 | D |
| GUE confirmed on 3-orders, 4-orders, sprinkled 2D/3D/4D | exp100 | D |
| Sub-Poisson r=0.12 was a PHASE-MIXING ARTIFACT (concatenation: r->0.39) | exp92, exp100 | D |
| Spectral compressibility chi->0 with proper unfolding | exp99 | D |
| GUE generic to ALL dense antisymmetric matrices (Erdos-Yau) | exp57 | D |
| GUE confirmed at N=1000 via sparse eigsh | exp75 | D |
| Number variance: GUE is short-range only (linear Sigma^2, not log) | exp54, exp56 | D |
| CDT c_eff~1, causet c_eff->inf (different universality classes) | exp44 | E |
| Kronecker product: C^T-C = A_T tensor J, n_pos=floor(T/2) (6-step proof) | exp91, exp107 | E |
| Exact eigenvalue: mu_k = cot(pi(2k-1)/(2T)), verified to 10^-14 | exp107 | E |
| Non-uniform extension: n_pos = floor(T/2) for ANY slice sizes | exp107 | E |
| Wightman depends ONLY on temporal indices (spatial variation ~10^-17) | exp107 | E |
| Single perturbation increases n_pos by 1 (extreme fragility) | exp107 | E |
| Kronecker generalizes to all spatial dimensions (d=1,2,3) | exp106 | E |
| S(N/2) depends only on T, independent of spatial size s | exp107 | E |
| W_T is exactly Toeplitz (time-translation invariant at lattice level) | exp107 | E |
| Eigenvalue sum rule: Σλ_k = (2/π)ln(T) + 0.527 (R²=1.000) | exp107 | E |
| CDT SJ vacuum is efficient MPS: bond dimension polynomial in T | exp107 | E |
| Continuum limit caveat: c_eff → 0 as T,s → ∞ (S bounded, ln N grows) | exp107 | E |
| 5% disorder doubles c_eff; 30% thinning preserves c=1 | exp85 | E |
| Fiedler value SATURATES at ~1.4-1.6 for N>100 (N^0.054, NOT N^0.32) | exp102 | F |
| Hasse diameter = 6 at N=3000, grows as N^0.11 (NOT Theta(sqrt(N))) | exp102 | F |
| Spectral embedding: 19 eigenvectors give R^2=0.83-0.91 | exp99 | F |
| Link fraction: perfectly monotonic BD order parameter (60% jump) | exp66 | F |
| Hasse diagram triangle-free (girth>=4), independence alpha~N^0.83 | exp86 | F |
| E[f]=1/2 with Var[f]~1/(9N) (proved, verified exactly) | exp61, exp64 | G |
| Antichain=2*sqrt(N) via Vershik-Kerov, Tracy-Widom fluctuations | exp61, exp64 | G |
| Master formula P[k|m] (2-order specific) | exp72, exp89 | G |
| E[links]=(N+1)H_N-2N, link fraction 2ln(N)/N + 2(γ-1)/N (undirected) | exp72, exp102 | G |
| E[S_Glaser]=1 for all N>=2 (universal benchmark) | exp80 | G |
| E[S_BD]=2N-NH_N (exact, verified) | exp80 | G |
| Limit shape f(alpha)=4[-(1+alpha)ln(alpha)+2alpha-2] | exp80 | G |
| Exact Z(beta) for N=4 (19 action levels) and N=5 (87 levels) | exp61 | G |
| Maximal elements E[max]=H_N, k-antichains E=C(N,k)/k! | exp103 | G |
| Chain-antichain symmetry: E[chains of length k] = E[antichains of size k] = C(N,k)/k! | exp103 | G |
| Unimodality: E[N_k] monotonically decreasing (mixture of linear decreasing functions) | exp103 | G |
| E[S_BD(epsilon)] exact formula verified to 1% for ε=0.05-0.30 | exp103 | G |
| Link fraction constant resolved: 2ln(N)/N (undirected), not 4ln(N)/N (directed) | exp103 | G |

## Cross-Paper Connections (Exp105)

| Connection | Papers | Strength | Key Evidence |
|---|---|---|---|
| Kronecker -> CDT spectrum | E -> D | **EXACT** (9.0) | Predicted spectrum matches to 10^{-15}. CDT is NOT GUE -- deterministic. |
| Master formula -> interval entropy | G -> A | **<2% error** (8.0) | Analytic H at beta=0 matches MCMC within 1.5-1.9%. |
| E[S_Glaser]=1 -> beta_c scaling | G -> A | **Constrains** (8.0) | sqrt(Var)*beta_c ~ O(1) recovers Glaser formula 1.66/(N*epsilon^2). |
| Kronecker -> CDT entanglement | E + B5 | **n_pos exact** (8.0) | S_ent predicted to ~75%. Gap from within-slice corrections. S depends only on T. |
| Antichain -> spatial extent | G -> B5 | **Quantitative** (7.5) | 2*sqrt(N) gives ~4 elements per region at N=50, explaining noise. |
| Gram identity universal | C -> all | **Exact** (8.5) | (-Delta^2)_ij = (4/N^2)*kappa_ij on ANY partial order via transitivity. |
| Link fraction -> ER=EPR | F -> C | **Partial r=0.42** (6.5) | Independent of density. Hasse structure controls |W|. |
| Phase-mixing -> all observables | D -> all | **Mild** (6.5) | 0.8-6.5% differences at N=30. Stronger at larger N. |

## 600-Idea Search Summary

| Ideas | Rounds | Best Score | Key Theme |
|---|---|---|---|
| 1-55 | Rounds 1-9 | 7.5 | SJ vacuum exploration (c~3, ER=EPR, quantum chaos, monogamy) |
| 56-100 | Rounds 10-12 | 7.0 | Deeper SJ (propagator decay, time asymmetry, Renyi, contour) |
| 101-150 | Rounds 13-17 | 7.5 | New observables (Fiedler, treewidth, info bottleneck, antichain thm) |
| 151-200 | Rounds 18-22 | 7.5 | Dimension encoding, BD transition, synthesis, wild cards |
| 201-250 | Rounds 23-27 | 7.5 | Path entropy, master interval formula, analytic proofs |
| 251-300 | Rounds 28-32 | 7.5 | SJ v2, physics connections, computational novelty, wild cards v2 |
| 301-400 | Rounds 33-42 | 7.5 | Large-N sparse, math proofs, Fiedler theory, KR deep dive, CDT mechanism |
| 401-500 | Rounds 43-52 | 7.5 | Paper strengthening, Kronecker theorem, sub-Poisson debunked, universality tests |
| 501-600 | Rounds 53-60 | 8.0 | Deep Kronecker (exact eigenvalues, Wightman factorization, fragility), large-N corrections (Fiedler saturation, diameter), universal Gram identity, higher-dim GUE |

**Why the ceiling holds at 8.0:** At N=30-200, three mechanisms defeat every idea: (1) density dominance -- most causet-vs-null differences trace to ordering fraction; (2) finite-size contamination -- corrections exceed target signals; (3) no exact finite-N predictions from causal set theory to test against. The Kronecker theorem (Paper E) achieves 8.0 because it provides exact analytic predictions verified to machine precision, bypassing all three limitations.

## Potential New Papers (from Exp93 evaluation)

| Paper | Score | Status | Key Data |
|---|---|---|---|
| **H: BD Transition Comprehensive Study** | 7.5 | STRONGEST CANDIDATE | 10+ observables: interval entropy (87% drop), link fraction (60% jump), Fiedler (243% jump), latent heat (extensive), nucleation (sharp), KR layers (~sqrt(N)), spectral stats (GUE both phases). No published paper has >3 observables. |
| **I: Graph-Theoretic Dimension** | 7.5 | FEASIBLE | 5 independent estimators (path entropy, Fiedler, treewidth, ordering fraction, interval entropy) across d=2-5. Path entropy linear in d (R^2>0.97). Gap at d>=4. |
| **J: SJ Vacuum Reproduces Physics** | 7.0 | FEASIBLE | Casimir 1/d (robust, CV=0.134), Newton ln(r) (wins at all N, coeff drifts), area law (S~A^0.345), conformal anomaly (<T>~R). Qualitative agreement, quantitative coefficients off. |
| **K: 600 Experiments Meta-Paper** | 6.5 | RISKY | Lessons learned + recommendations. Needs honest framing. |

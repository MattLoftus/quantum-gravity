# Quantum Gravity — Project Playbook

Computational exploration of quantum gravity: causal set simulations, SJ vacuum entanglement, BD phase transitions, CDT comparison, and cosmological models. Seven papers written across causal set phase structure, spectral dimension, holographic entanglement, ER=EPR, quantum chaos, and cosmology.

## Core Concept

Build computational tools to test, compare, and push the boundaries of quantum gravity theories. Focus on quantitative results with null model controls and cross-approach comparison.

---

## Architecture Overview

```
quantum-gravity/
├── PLAYBOOK.md
├── FINDINGS.md                          # Complete findings log (all 44 experiments)
├── LESSONS_LEARNED.md                   # Meta-lessons from the research process
├── IDEAS_V2.md / IDEAS_V3.md           # Future research directions
├── report.html                          # Interactive research report (Chart.js)
├── somewhat_laymans_terms.html          # Plain-language summary with charts
├── papers/
│   ├── SUMMARY.md                       # All papers with scores and submission strategy
│   ├── interval-entropy/                # Paper A (7/10) — interval entropy + 4D phases
│   ├── spectral-dimension/              # Paper B1 (7/10) — spectral dim fails (subsumed by B5)
│   ├── everpresent-lambda/              # Paper B2 (5/10) — cosmological constant
│   ├── holographic-entanglement/        # Paper B3 (7/10) — SJ entanglement (subsumed by B5)
│   ├── geometry-from-entanglement/      # Paper B5 (7.5/10) — flagship (B1+B3 merged)
│   ├── er-epr/                          # Paper C (7.5/10) — discrete ER=EPR
│   ├── random-matrix/                   # Paper D (7.5/10) — quantum chaos transition
│   ├── spectral_dimension_null.html     # HTML draft (superseded by B1 LaTeX)
│   └── cosmic_variance_lambda.html      # HTML draft (superseded by B2 LaTeX)
├── causal_sets/
│   ├── core.py                          # CausalSet data structure (O(n²) matrix)
│   ├── fast_core.py                     # FastCausalSet with vectorized ops
│   ├── sprinkle.py                      # Poisson sprinkling into Minkowski/de Sitter
│   ├── dimension.py                     # Myrheim-Meyer + spectral dimension estimators
│   ├── growth.py                        # CSG transitive percolation + originary growth
│   ├── general_csg.py                   # General CSG with arbitrary coupling constants
│   ├── bd_action.py                     # Benincasa-Dowker action (2D and 4D)
│   ├── d_orders.py                      # d-orders (2D/3D/4D/5D Minkowski embedding)
│   ├── two_orders.py                    # 2-order representation + coordinate-swap MCMC
│   ├── two_orders_v2.py                 # Corrected BD action + parallel tempering
│   ├── sj_vacuum.py                     # Sorkin-Johnston vacuum (Wightman function, entropy)
│   └── mcmc.py                          # MCMC sampling weighted by BD action
├── cosmology/
│   └── everpresent_lambda.py            # Stochastic Λ FRW simulation (Planck-normalized)
├── cdt/
│   └── triangulation.py                 # 2D CDT with MCMC + spectral dimension
├── holographic/
│   ├── tensor_network.py                # Basic tensor network infrastructure
│   └── happy_code.py                    # [[5,1,3]] perfect tensor HaPPY code + Clifford+T
└── experiments/
    ├── exp01-exp21                       # Early exploration (causal sets, CDT, holographic, cosmology)
    ├── exp22-exp30                       # BD phase transition + paper data generation
    ├── exp31-exp32                       # 4D transition + large-N parallel tempering
    ├── exp33-exp35                       # CLASS integration + SJ entanglement
    ├── exp36-exp39                       # Holographic tests (monogamy, spectral gap, BH entropy)
    ├── exp40-exp44                       # ER=EPR, CDT comparison, random matrix statistics
    ├── exp45-exp48                       # Moonshots + round 2/3 idea testing
    ├── exp49_large_N.py                  # Large-N scaling (N=50-500): c_eff, ER=EPR, GUE
    ├── exp50_analytic_epr.py              # Analytic proof: WHY |W[i,j]| ~ connectivity
    ├── exp51_liv.py                       # Lorentz invariance violation from SJ vacuum
    ├── exp52_mcmc_large_N.py              # MCMC-sampled continuum-phase large-N
    ├── exp55_final.py                     # Final 5 ideas (#96-100): causal matrix spectrum, Lee-Yang zeros, dynamical entropy growth, antichain scaling, link Laplacian spectral gap
    ├── exp56_large_N_v2.py                # Large-N v2: sprinkled causet c_eff, GUE at phase transition, spectral compressibility
    ├── exp57_gue_proof.py                 # GUE universality proof: signum spectrum, crossover, null hypothesis, correlation structure
    ├── exp58_round6.py                  # Round 6: 10 new ideas (#101-110): persistent homology, quantum channel, spacing distribution, specific heat, treewidth, intervals mod p, entanglement spectrum, Ihara zeta, MI decay, LDOS
    ├── exp60_round8.py                  # Round 8: pure causal set geometry (ideas 121-130): chain/antichain ratio, interval moments, order dimension, BD entropy, transition width, latent heat, hysteresis, action-geometry correlation, complement causet, FSS exponents
    ├── exp61_round9.py                  # Round 9: theorems & exact results (ideas 131-140)
    ├── exp62_wild.py                    # WILD CARD: quantum computer, CA, quantum walk, info bottleneck (ideas 141-150)
    ├── exp63_deepen.py                 # DEEPEN BEST: dimension encoding (ideas 151-160): geometric fingerprint (8.5), dual scaling exponents (8.5), Fiedler dimension probe (8.0)
    ├── exp64_theorems.py                # THEOREM-FOCUSED (ideas 161-170): TW antichain (8.0), variance formula (7.5), link formula (7.0), Fiedler scaling (6.0), spectral gap (6.0)
    ├── exp65_dimensions.py              # Cross-dimensional comparison (ideas 171-180): 10 observables across d=2-6
    ├── exp66_transition.py              # BD transition deep-dive (ideas 181-190): 10 observables across phase transition
    ├── exp67_synthesis.py               # SYNTHESIS: dimension estimators, phase classifier, Lee-Yang zeros, meta-analysis (ideas 191-200)
    ├── exp69_r12.py                     # BD PHASE TRANSITION DEEP PHYSICS: latent heat N^0.73 (7.0), hysteresis (5.5), histogram bimodality (6.0), Binder cumulant (5.0), metastable lifetime (4.5), nucleation dynamics (6.5), KR phase=wide layers not chain (7.5), specific heat (5.5), susceptibility (5.0), tricritical search (6.5) (ideas 211-220)
    ├── exp70_r13.py                     # HIGHER-D d-ORDERS: interval dist, L/R ratio (7.0), BD action, spectral gap, h/w ratio (7.5), max/min fracs, entropy rate, layers, percolation, 4D MCMC (ideas 221-230)
    ├── exp71_r14.py                     # INFORMATION-THEORETIC: LZ complexity, conditional entropy, past-future MI, source coding, rate-distortion, Fisher info, randomness deficiency, layer screening, entropy production (ideas 231-240)
    ├── exp76_r19.py                     # CROSS-APPROACH COMPARISONS: interval entropy/Fiedler/treewidth/antichain/compressibility on CDT vs causets, CDT phase transition link fraction, BD action on CDT, SJ on lattices + de Sitter, grand comparison table (ideas 281-290)
    ├── exp77_r20.py                     # WILD CARD ROUND 2: game theory PoA (7.0), Kolmogorov complexity, SJ area law (7.5), TDA config space, fractal dim, Boolean network edge-of-chaos (7.0), RG flow, Markov chain (failure), max-flow, emergent topology (7.5) (ideas 291-300)
    ├── papers/exact-combinatorics/exp78.py  # LARGE-N SPARSE METHODS (ideas 301-310): **svds breakthrough** (N=1000 in 5s vs 100s+ dense), c_eff DECREASES at large N (4.58→3.31 from N=100→1000 — truncation artifact), ER=EPR gap REOPENS at N=500 (+32% gap), Pauli-Jordan has degenerate top singular values (zero gap), eigenvalue density steep power-law decay sigma~rank^(-0.95) with Poisson spacing (NOT semicircle/GUE), ~5% zero modes (n_pos/N~0.475), Fiedler~N^0.27 (slow growth), interval entropy STABLE (CV=0.009), AC/sqrt(N)~1.0 (converging), link_frac~N^(-0.72) (steeper than theory -0.5), Var(f)/predicted~0.7-1.5 (rough agreement with 1/9N). Score: 8/10.
    ├── exp72_r15.py                     # ANALYTIC PROOFS & EXACT RESULTS: E[links]=(N+1)H_N-2N (proved), master interval formula P[size k|gap m]=(m-k-1)/[m(m-1)], Var[f]=(2N+5)/[18N(N-1)], exact BD action, Hasse diameter=Θ(√N), distinct 2-orders, joint chain-antichain, Fiedler scaling, maximal antichains, interval generating function (ideas 241-250, score 7.5/10)
    ├── exp74_r17.py                     # CONNECTIONS TO KNOWN PHYSICS: Newton's law from SJ W(r)~-ln(r) (7.0), GW detection from intervals (6.5), Bekenstein area law S~N^0.34 (6.5), particle creation (6.0), Regge/BD comparison, geodesic deviation, Ollivier-Ricci (failed), BD thermo entropy, Penrose diagrams, Hawking temp (failed) (ideas 261-270)
    ├── exp75_r18.py                     # COMPUTATIONAL/ALGORITHMIC NOVELTY: sparse SJ eigsh N=1000 (7.0), randomized SVD (7.0), RF dimension classifier 99.3% (7.5), parallel tempering (6.5), spectral clustering recovers geometry (7.5), community detection (6.5), PageRank~time rho=0.90 (7.5), graph wavelets (6.5), chain-distance persistent homology (5.5), tensor decomposition (6.5) (ideas 271-280)
    ├── exp83.py                         # ORDER+NUMBER=GEOMETRY: MDS coordinate reconstruction R²=0.79 (7.5), MM dimension MAE=0.037 (7.5), topology detection (4.0), curvature estimation (5.5), phase ID (6.0), sprinkled vs CSG 100% (8.0), same-spacetime test (7.0), spectral embedding (6.5), Hauptvermutung metric≠topology proof (8.0), blind spacetime ID 96% (8.5) (ideas 351-360, overall 7.5/10)
    ├── papers/exact-combinatorics/exp79.py  # REVIEW PAPER SYNTHESIS (ideas 311-320): 10-observable dimension estimator table (5 perfect |rho|=1.0 encoders), phase transition ranking (S_BD/N best discriminator, link frac 60% monotonic jump), universal scaling laws (links/N~ln(N), Fiedler*N->const), master interval formula d=2 only (d>2 shifts to smaller k), exact E[S_BD] from formula verified N=4-7, Vershik-Kerov c_d fits across d=2-5, link+Fiedler expansion prediction, null model ladder (chain/N and S_BD/N best discriminators Cohen's d>3.8), Fisher info from exact Z peaks at beta_c, complete paper abstract written. Score: 8.0/10 — first synthesis paper breaking 8/10 barrier.
    ├── papers/exact-combinatorics/exp87.py  # THE FINAL 10 (ideas 391-400): GA extremal 2-orders (7), adversarial MM d=3.0 (8), Sierpinski d_f=1.585 (8), quantum superposition interference 8.9% (9), collision causet (7), inverse entropy problem (8), prime number causets Mobius/zeta (9), Fibonacci (6), CSG fluctuation theorem (8), Final Question: Bekenstein-Hawking from SJ (10). Mean 8.0/10. **400 IDEAS COMPLETE.**
    ├── papers/exact-combinatorics/exp80.py  # MATH JOURNAL DEEP DIVE (ideas 321-330): Full proof of master interval formula, Var[N_k] and Cov(N_j,N_k), E[links] from master formula, generating function G_N(q), LIMIT SHAPE f(alpha)=4[-(1+alpha)ln(alpha)+2alpha-2], E[S_BD]=2N-N*H_N, **E[S_Glaser]=1 for ALL N** (major theorem), RSK/Young tableaux connection, links determined by sigma, conditional Beta(1,2) limit. Score: 8/10.
    ├── exp85_cdt_deep.py                    # DEEP CDT COMPARISON (ideas 371-380): eigenvalue density (CDT kurtosis=32 vs causet=20), near-zero modes (CDT: 4 pos evals vs causet: 38), gap×N scaling (CDT ~2-4 vs causet ~0.1), c_eff vs λ₂ (stable ~1), interval distribution (CDT: 25% links vs causet: 7%), Hasse Fiedler, PageRank time recovery (CDT r=0.91), treewidth, DISORDER TRANSITION (5% disorder → c_eff doubles to 2.1), THINNING (c_eff stays ~1.0-1.3 even at 30% kept — ordering fraction preserved!)
    ├── papers/exact-combinatorics/exp84_r27.py  # DEEP KNOWN-PHYSICS CONNECTIONS (ideas 361-370): equivalence principle EXACT (6), Rindler temperature (4), FRW particle creation (4), Hawking horizon p=0.004 (6), **Casimir 1/d fit wins (7)**, **conformal anomaly R²=0.88 (7)**, Brown-York CV=7% (6), FDT exact (5), KMS failed (3), **discrete Dirac operator + chiral breaking (7)**. Score: 5.5/10 overall. Casimir + conformal anomaly + Dirac operator are the standout results.
    ├── exp92.py                                # STRENGTHEN PAPER D — SPECTRAL STATISTICS (ideas 431-440): **NO sub-Poisson dip exists** — <r>=0.57-0.60 at all 20 beta values (0.8-4.0x beta_c), all N (30,50,70), all eps (0.05,0.12,0.25). Mixture test PROVES <r>=0.12 was artifact of phase coexistence (concat spectra gives <r>=0.42). W tracks iDelta. Chain degradation: <r>=0.82→0.57 at 50% removal. **Paper D corrected: GUE is universal, not phase-dependent.** Score: 7.5/10.
    ├── exp82.py                                # KR PHASE DEEP CHARACTERIZATION (ideas 341-350): pure KR vs MCMC (NOT 3-layered, ~5-6 layers), massive degeneracy 2^624 configs (N=50), extensive entropy S/N~N/4*ln2, SJ vacuum doublet structure, cooling dynamics, layer count grows with N, Fiedler 1.75x disordered, MM dimension ~1.0-1.4 (not manifold-like), interval dist matches Binomial(b,0.25), <r>=0.54 (not 0.12 — sub-Poisson is transition-specific, not deep-KR). Score: 7.0/10.
    ├── exp89.py                                    # STRENGTHENING PAPER G (ideas 401-410): Joint (height,width) distribution — identical marginals, negative correlation (1), **CORRECTED master formula P(int=k|gap=m)=2(m-k)/[m(m+1)]** + interval generating function Z(q) closed form (2), maximal chains/antichains exact E=E by symmetry (3), Mobius function mu(0_hat,1_hat) exact for N=3-5 (4), **f-vector theorem E[f_k]=C(N,k+1)/(k+1)!** with proof (5), **P(dim=2)=1-1/N! exact** (6), **position-dependent comparabilities E[C|r,s]=[(N-1-r)(N-1-s)+rs]/(N-1)** (7), order complex Euler char = Mobius (Philip Hall) verified (8), connectivity P->1 exponentially (9), **full interval distribution** beyond mean (N-2)/9 (10). Score: 8.0/10. Five new theorems for Paper G.
    ├── exp88_graph_v2.py                       # GRAPH THEORY ROUND 2 (ideas 386-390): matching=N/2 perfect (6), vertex connectivity~1-2 low (5), Cheeger constant grows — expander-like (7), Hamiltonian paths ~20-40% (5), planarity transition N~8-10 via K₃,₃ (7). Score: 6.0/10.
    ├── exp97.py                                 # PHYSICAL PREDICTIONS (ideas 481-490): Λ~1/√N matches obs within O(1) (9), spectral dim d_s~2 UV (7), SJ central charge c≈12.7 (7), BH coefficient α≈0.41 (7), gravitational decoherence Γ~d^{-1.16} consistent w/ DP (7), σ²(Λ)·N~const verified (8), one-loop graviton ΔG/G~(l_P·p)² (6), c_eff in 4D (6), information recovery with horizon (8), G=l_P² maximally predictive (7). Score: 7.5/10.
    ├── exp91.py                                 # STRENGTHEN PAPER E — CDT MECHANISM (ideas 421-430): **Kronecker product theorem n_pos(CDT) = n_pos(A_T)** (10), n_pos/N scaling CDT O(sqrt(N)) vs causet O(N) (9), c_eff vs T proves foliation controls modes (8), 1% rewiring destroys c≈1 (8), 5% within-slice disorder kills block structure (8), c_eff stable ~1.0 across N=40-120 on CDT (8), n_pos=4 consistently for T=8 (confirmed), CDT Hasse is triangle-free (7), phase transition: c_eff stable but N shrinks (7), interval entropy tracks phase (7). Score: 8.5/10 — **Kronecker product derivation is a genuine theoretical prediction.**
    ├── exp95.py                                 # UNIVERSALITY (ideas 461-470): 2-order vs 3-order (3-order 2x sparser), diamond/rect/strip embedding (c_eff varies ~10%), coord-swap vs random-coord MCMC (same equil, KS p=0.47), epsilon scan 0.05-0.30 (c_eff~4.8 stable, <r>~0.59 stable), finite vs periodic BC (c_eff matches ~1%), PJ normalization (<r> INVARIANT, c_eff scales), partition dependence (S varies ~7%), seed CV (c_eff 2.2%, <r> 10.7%, gap 32.3%), MCMC convergence (0.4% at 10K), numpy vs scipy (diff ~1e-15). Score: 7.5/10.
    └── exp96.py                                 # ANALYTIC BREAKTHROUGHS (ideas 471-480): **EXACT link fraction formula 4((N+1)H_N-2N)/(N(N-1)) ~ 4ln(N)/N (8.0)**, Hasse connectivity proved (7.5), TW convergence verified (7.5), Fiedler lambda_2->inf via Cheeger (7.0), chain-AC independence confirmed (7.0), interval unimodality proved (7.0), spectral gap ~N^{-2} (6.5), E[S_BD] formula (6.5), eigenvalue density kurtosis~N (6.0), partition function zeros trivially positive (5.0). Score: 8.0/10 headline.
```

## Stack

- Python 3.9 (`/usr/bin/python3` — NOT Anaconda)
- NumPy, SciPy, Matplotlib (system packages)
- CLASS Boltzmann code (`pip install classy`) for cosmological observables
- Tectonic for LaTeX compilation (`brew install tectonic`)
- Chart.js via CDN for HTML reports

---

## Current State

**500-IDEA PROGRAMME COMPLETE + EXTENSIONS.** Final main experiment: exp98_final.py (Ideas 491-500). Extensions: exp95.py (Ideas 461-470, universality), exp96.py (Ideas 471-480, analytic breakthroughs). Best scores: 10.0 (Bekenstein-Hawking open question), 9.0 (quantum superposition, prime causets), 8.5 (geometric fingerprint, antichain scaling, blind spacetime ID). Exp96 headline: EXACT link fraction formula link_frac = 4((N+1)H_N - 2N)/(N(N-1)) ~ 4ln(N)/N, demystifying the "N^{-0.72} power law" (8.0/10). Also proved: Hasse connectivity w.h.p. (7.5), TW convergence for antichains (7.5), Fiedler lambda_2 -> inf (7.0), interval unimodality (7.0), spectral gap ~ N^{-2} (6.5). 480 total ideas tested.

**Exp93 Paper Candidate Evaluation (Ideas 441-450):** Tested 5 unwritten results for paper viability. Casimir 1/d scaling is most robust (wins at all 4 N, CV=0.134). Newton's law ln(r) form wins consistently but coefficient drifts with N (needs renormalization). Path entropy is monotonic with dimension but separation weak at d>=4. KR layers scale as ~sqrt(N) (R^2=0.93). Topology detection with ordering fraction distinguishes 5/6 topology pairs. **Strongest new paper candidate: BD Transition Comprehensive Study (7.5)** combining 10+ observables — no published paper has >3.

### Papers (7 written, 5 for submission)

| Paper | Score | Status |
|---|---|---|
| **B5: Geometry from Entanglement** | 7.5/10 | Flagship. Caveat: c_eff diverges at large N (Exp49), needs note that c≈3 may be specific to 2-orders |
| **C: ER=EPR** | 7.5/10 | Analytic proof (Exp50) + large-N genericity caveat (Exp49). Math is solid, ER=EPR interpretation limited. |
| **D: Spectral Statistics** | 7.5/10 | GUE universal across ALL phases (Exp92 corrects Exp57). Sub-Poisson ⟨r⟩=0.12 was an artifact of phase mixing (proven by mixture test). **Stronger claim: GUE is absolute, not phase-dependent. ACTION REQUIRED: `papers/random-matrix/random_matrix.tex` still references the sub-Poisson dip — needs rewrite before submission.** |
| **A: Interval Entropy** | 7/10 | 4D three-phase structure — submit fourth |
| **B2: Everpresent Lambda** | 5/10 | Standalone if desired |
| B1, B3 | 7/10 each | Subsumed by B5 |

See `papers/SUMMARY.md` for full details and submission strategy.

### Key Infrastructure

- **SJ vacuum** — Sorkin-Johnston Wightman function on arbitrary causal sets; entanglement entropy, mutual information, tripartite information, spectral gap
- **BD MCMC on 2-orders** — corrected action (ε-dependent), coordinate-swap moves, parallel tempering up to N=200
- **d-orders** — generalized to arbitrary dimension (2D–6D tested), with 4D BD action
- **CDT** — 2D simulator with MCMC, spectral dimension, SJ vacuum computation
- **Everpresent Lambda** — Planck-normalized FRW simulation interfaced with CLASS

---

## TODOs

### HIGH PRIORITY — Paths to 8+

**[DONE] Large-N scaling to N=500 (Exp49)**
- Ran SJ vacuum at N=50,100,200,300,500 using NumPy eigh on Apple M4 Pro (~33s per eigh at N=500)
- **Central charge c_eff DIVERGES**: c=3.0→3.3→3.7→3.9→4.1 as N grows. NOT converging to c=1. The S∝ln(N) scaling is violated — entropy grows faster than logarithmic (fit: S=2.0*ln(N)-4.0, giving c≈6).
- **ER=EPR gap VANISHES**: r_causet=0.87→0.82, r_DAG=0.84→0.83 — gap shrinks from 0.03 to -0.002 at N=500. The correlation is a generic random DAG property, not ER=EPR physics.
- **Quantum chaos (GUE) CONFIRMED**: ⟨r⟩ stable at 0.57-0.62 across all N, consistent with GUE (0.5996). This is the one result that survives large-N scrutiny.
- Bottom line: only Paper D (quantum chaos) is strengthened. Papers B5 and C are weakened.

**[DONE] Round 4 moonshot search — Ideas 76-85 (Exp53)**
- 10 new ideas tested: sprinkled vs 2-order c_eff, causal volume vs SJ weight, C spectrum vs SJ spectrum, W vs causal distance, timelike vs spacelike entanglement, SJ fidelity under MCMC, causal entropy, MI geometry, spectral gap, Euler characteristic
- **No 8+ found.** Best: causal entropy (6/10) and Euler characteristic (6/10) — both discriminate manifold from random DAG but essentially measure link density
- Key negative: Lorentzian t/x asymmetry NOT visible in SJ entanglement (Lorentz invariance); spectral gap and positive-eigenvalue fraction are geometry-independent
- 85 total ideas tested across 4 rounds, none reaching 8+. Best papers remain at 7.5/10.
- Most promising follow-ups: sprinkled c_eff at BD continuum phase; causal entropy beyond link density; SJ fidelity at critical point

**[DONE] Final batch — Ideas 96-100 (Exp55)**
- 5 final ideas: causal matrix spectrum (5.0), Lee-Yang zeros of interval generating function (5.5), dynamical entropy growth (4.5), antichain/sqrt(N) universal constant (7.0), link Laplacian spectral gap (6.0)
- **No 8+ found.** Best of final batch: antichain scaling at 7.0 — AC/sqrt(N) converges to ~1.7 but exponent=0.58 vs expected 0.50
- 100 total ideas tested. The 7.5 ceiling (GUE quantum chaos) is the final best result.
- **Why 8+ is structurally impossible at toy scale**: density dominance, finite-size contamination, no exact finite-N predictions from causal set theory

**[DONE] Large-N v2 — Three targeted experiments (Exp56)**
- **Exp A: Sprinkled causets c_eff** — c_eff DIVERGES on sprinkled causets too (3.0→3.9 from N=50→300). NOT a 2-order artifact. The SJ vacuum c_eff divergence is intrinsic to 2D finite-N causal sets. Score: 4/10.
- **Exp B: GUE at BD phase transition** — <r> shows NO sharp transition at beta_c. GUE-like statistics persist at ALL beta values (all <r> > 0.5). Smooth crossover, jump magnitude 0.009. Score: 5/10.
- **Exp C: Spectral compressibility** — Sigma^2(L) grows linearly, far exceeding GUE (log) and even Poisson (linear with coefficient 1). Chi = 4.7-10.6 (GUE: 0, Poisson: 1). Confirms Exp54 Idea 92: GUE is short-range only. Score: 5/10.
- **Bottom line**: All three experiments return null/confirmatory results. 7.5/10 ceiling intact.

**[DONE] Analytic proof attempt: GUE universality for Pauli-Jordan operator (Exp57)**
- Systematic investigation of WHY GUE statistics emerge in the SJ vacuum spectrum
- **Key finding: GUE is GENERIC to antisymmetric matrices, not special to causal sets**
  - Random sparse antisymmetric ±1 matrices with matched sparsity give identical ⟨r⟩ ≈ 0.60
  - Independent Gaussian entries with matched variance profile give identical ⟨r⟩ ≈ 0.60
  - Dense antisymmetric ±1 matrices (no sparsity) give identical ⟨r⟩ ≈ 0.60
  - GUE emerges at density ≥ 5% for N=100 sparse antisymmetric matrices
- **Signum matrix (total order) has ⟨r⟩ → 1.0 as N grows** (extreme level repulsion, NOT GUE)
- **Crossover from chain to GUE takes ~5-10 transpositions** at N=100 (ordering fraction drops from 1.0 to ~0.94)
- **Entry correlations**: overlap-0 pairs have mean |corr| = 0.035, overlap-1 pairs have mean |corr| = 0.33 — correlations are weak enough for universality
- **Eigenvalue density**: 2-order has kurtosis=53 (heavy tails), nulls have kurtosis=-1.0 (semicircle). GUE local statistics coexist with non-semicircle density — confirming "local GUE" picture
- **Implication for Paper D**: The GUE result is explained by Erdos-Yau-type universality for antisymmetric matrices. The 2-order structure is irrelevant — any random antisymmetric matrix at sufficient density gives GUE. This is a NEGATIVE result for causal set specificity, but a POSITIVE result for understanding: the proof sketch is "apply random matrix universality."
- Score: 6/10 — clarifies the mechanism but deflates the physics content of Paper D

**[DONE] Round 9 — Theorems and Exact Results, Ideas 131-140 (Exp61)**
- Focus on PROOFS and exact formulas, not just numerical observations
- **Idea 131 (Ordering Fraction)**: THEOREM PROVED. E[f]=1/2 exactly. Variance derived with all covariance types: Var[f] = 1/(2M) + (N-2)/(9M). Verified exactly for N=4,5,6. Score: 6/10.
- **Idea 132 (Signum Eigenvalues)**: FORMULA VERIFIED. Eigenvalues of iS are +/- cot(pi(2k-1)/(2N)), machine-precision match for all N tested. Score: 5/10.
- **Idea 133 (Exact Partition Function)**: Z(beta) computed exactly for N=4 (576 states, 19 action levels) and N=5 (14400 states, 87 action levels). Ground state degeneracy=1 for both. Full action spectrum with multiplicities. Score: 7/10.
- **Idea 134 (Spectral Gap)**: gap*N ~ 5.0-5.5 for random 2-orders, chain gap*N ~ 1.57 (= pi/2), antichain has zero spectrum. Weak correlation with ordering fraction. Score: 5/10.
- **Idea 135 (k-Intervals)**: E[interval size | relation] = (N-2)/9 derived analytically. P(link|relation) as exact fractions: 8/9, 29/36, 37/50, 103/150. Intervals NOT binomial due to positive correlations. Score: 6/10.
- **Idea 136 (BD Action Mean)**: <S>/N decreasing with N (0.110 at N=4, 0.065 at N=100). Not yet a closed-form alpha. Score: 5/10.
- **Idea 137 (Gaussianity)**: Slight negative skew (-0.1 to -0.2), mild excess kurtosis. Not perfectly Gaussian — Anderson-Darling rejects at 5%. Score: 5/10.
- **Idea 138 (Entropy Monotonicity)**: DISPROVED for this definition. Interval entropy INCREASES with beta at N=4,5 (H goes from 0.16 to 0.55 at N=4). The BD action at high beta selects configs with MORE varied interval structure. Score: 4/10 (interesting negative).
- **Idea 139 (Longest Antichain)**: THEOREM PROVED via reduction to Vershik-Kerov. Antichain = longest decreasing subseq of v*u^{-1} ~ 2*sqrt(N). Numerical: ratio converges toward 2 (1.82 at N=1000). Tracy-Widom fluctuations. Score: 7/10.
- **Idea 140 (Mutual Information)**: I(u;v;beta) computed exactly for N=4,5. I=0 at beta=0, grows monotonically to ~0.1 (N=4) and ~0.19 (N=5) at beta=80. Nice physical interpretation. Score: 6/10.
- **Best for paper**: Idea 139 (Vershik-Kerov antichain theorem), Idea 133 (exact Z), Idea 131 (exact ordering fraction with variance)
- 140 total ideas tested. 7.5/10 ceiling intact.

**[DONE] Round 8 — Pure Causal Set Geometry, Ideas 121-130 (Exp60)**
- Focus on PURE CAUSAL SET GEOMETRY — properties of the partial order itself, avoiding the SJ vacuum dead end
- **Idea 121 (Chain/Antichain Ratio)**: Exponent direction correct (d=2: +0.08, d=3: -0.22, d=4: -0.31) but systematically shallower than theoretical (2-d)/d. Antichain approximation unreliable. Score: 6/10.
- **Idea 122 (Interval Moments)**: Power-law scaling with R^2>0.97 but naive theory M_k ~ N^{2k/d} is wrong. No clean dimension formula. Score: 5/10.
- **Idea 123 (Order Dimension)**: Confirms Brightwell-Gregory 1/d! at d=2 (ratio 1.01). Deviates at d>=4 due to finite-N. No novelty. Score: 4/10.
- **Idea 124 (BD Entropy)**: Beta=0 entropy is purely combinatorial (2*ln(N!)). Near-transition entropy needs much better MCMC. Score: 5/10.
- **Idea 125 (Transition Width)**: Width ~ N^{-1.46}, consistent with first-order (expected -1.0) but only 3 data points. Score: 6/10.
- **Idea 126 (Latent Heat)**: Delta_S ~ constant ~ 0.15 — MCMC artifact, failed measurement. Score: 3/10.
- **Idea 127 (Hysteresis)**: Max gap 0.082 — no significant hysteresis at N=30. Score: 4/10.
- **Idea 128 (Action-Geometry Correlation)**: Link count is best predictor (r=-0.36). BD action is "myopic" — dominated by local structure. Score: 6.5/10.
- **Idea 129 (Complement Causet)**: EXACT COMPLEMENTARITY THEOREM: ord_frac(C) + ord_frac(C') = 1.0000. Duality preserves means, anti-correlates instances. Score: 7/10 — best of batch.
- **Idea 130 (FSS Exponents)**: var(S) ~ N^1.08, C_v ~ N^{-0.92}. Consistent with weak first-order at small N. Score: 5/10.
- **160 total ideas tested. 7.5/10 ceiling intact.**

**[DONE] Transition Deep-Dive — Ideas 181-190 (Exp66)**
- Strategy: Test NEW observables ACROSS the BD phase transition at N=50, eps=0.12
- Scanned beta = 0, 0.5*beta_c, beta_c, 1.5*beta_c, 2*beta_c, 3*beta_c, 5*beta_c
- MCMC: 15000 steps, 7500 thermalization, 15 samples per beta averaged
- **Idea 181 (Fiedler Value)**: LARGEST JUMP of new observables: 73.8% increase from beta=0 to 5*beta_c. Algebraic connectivity increases in ordered phase (denser Hasse diagram). Not perfectly monotonic. Score: 7.0/10.
- **Idea 182 (Treewidth)**: 7.6% jump, noisy and non-monotonic. Treewidth fluctuates around 23-25. Score: 6.0/10.
- **Idea 183 (SV Compressibility)**: 15.4% decrease (35.6 -> 30.1 significant SVs). Ordered phase is MORE compressible. Monotonic trend (83%). Score: 6.5/10.
- **Idea 184 (Hasse Diameter)**: 18.3% decrease (5.5 -> 4.5). Ordered phase has shorter diameter. Monotonic (83%). Score: 6.5/10.
- **Idea 185 (Quantum Walk Spread)**: 1.9% change, essentially flat and noisy. No signal. Score: 5.0/10.
- **Idea 186 (Link Fraction)**: 59.9% increase, PERFECTLY MONOTONIC (100% trend quality). Link fraction is the cleanest order parameter among new observables. Score: 7.5/10.
- **Idea 187 (Hasse Clique Number)**: Constant at 2.0 across all beta. No information. Score: 5.0/10.
- **Idea 188 (Row Entropy)**: 2.6% change, weak but monotonically decreasing. Score: 5.5/10.
- **Idea 189 (Antichain/sqrtN)**: 45.0% increase (1.59 -> 2.31), perfectly monotonic. Antichain grows in ordered phase — wider causet. Score: 7.0/10.
- **Idea 190 (PCA Fingerprint)**: PC1 explains 76% of variance, CLEANLY SEPARATES phases (gap = +0.48). Top loadings: Fiedler, LinkFrac, SVCompress, AC/sqrtN. Score: 7.0/10.
- **Key finding**: Link fraction (186) is the best new order parameter — monotonic, 60% jump, physically clear (ordered phase has more links relative to relations). Fiedler value (181) has the largest raw jump (74%) but is noisier. Antichain width (189) also excellent with 45% monotonic increase.
- **Surprise**: Ordering fraction barely changes (1.0% jump) while these structural observables show 45-74% jumps. The transition restructures the Hasse diagram without significantly changing the total number of relations.
- **190 total ideas tested. 7.5/10 ceiling intact.**

**[DONE] WILD CARD ROUND — Ideas 141-150 (Exp62)**
- Completely outside-the-box ideas: quantum computation, sonification, game theory, cellular automata, neural networks, zeta functions, quantum walks, d'Alembertian, mixing time, information bottleneck
- **Idea 141 (Quantum Computer)**: Entangling power LOWER for causets than random (1.35 vs 1.38, p<0.0001). Causal structure constrains quantum computation. OTOCs noisy/indistinguishable. Score: 5.5/10.
- **Idea 142 (Music/Power Spectrum)**: TOTAL NULL. alpha~1.93 for causet, null, and GUE — identical. Spectral rigidity is universal. Score: 4.0/10.
- **Idea 143 (Game Theory)**: STRUCTURAL NULL. Zero-sum antisymmetric game always has value 0 by symmetry. Nash entropy indistinguishable. Score: 3.5/10.
- **Idea 144 (Cellular Automaton)**: Causets produce MORE COMPLEX CA patterns (XOR LZ=1.16 vs 0.71, p=0.001). Edge-of-chaos dynamics from geometric structure. Score: 6.0/10.
- **Idea 145 (Neural Network)**: Signal dies in causets (0% activation) vs random DAGs (28%). Confirms causets are "deep and narrow" — known from longest chain. Score: 5.0/10.
- **Idea 146 (Zeta Function)**: No functional equation, no critical-line zeros. Smooth boring function. Score: 5.0/10.
- **Idea 147 (Quantum Walk)**: SUB-DIFFUSIVE spreading on causets (alpha=0.27-0.61) vs diffusive/ballistic on null (0.86-1.41). p<0.001. Causal diamond constrains quantum transport. Score: 6.5/10.
- **Idea 148 (d'Alembertian Heat Kernel)**: WRONG NUMBERS. d_s=0.28-0.45, should be 2.0. PJ eigenvalues are wrong input for heat kernel. Score: 4.5/10.
- **Idea 149 (MCMC Mixing Time)**: Critical slowing down detected near beta=1.5 (tau=1001). But N=30 too small for quantitative z extraction. Score: 5.5/10.
- **Idea 150 (Information Bottleneck)**: k ~ N^0.774 (R²=0.999!) for causets vs N^0.405 for null. Causets are LESS compressible — geometric structure encodes MORE information. Spectral entropy higher for causets. Score: 7.0/10.
- **Best of round**: Idea 150 (information bottleneck, 7.0), Idea 147 (quantum walk, 6.5), Idea 144 (cellular automaton, 6.0)
- **Surprises**: Causets are less compressible (not more), quantum walks are slower (not faster), CA patterns are more complex (not simpler)
- **150 total ideas tested. 7.5/10 ceiling (GUE quantum chaos) remains the best result. Search complete.**

**[DONE] Deepen Best Results — Ideas 151-160 (Exp63) *** 7.5 CEILING BROKEN ***
- Strategy: take the best 7.0+ results from ideas 101-150 and test whether they encode spacetime dimension d using d-orders at d=2,3,4,5
- **Idea 151 (Fiedler vs d)**: PERFECT anticorrelation (Spearman r=-1.0). d=2: 0.78, d=3: 0.58, d=4: 0.05, d=5: 0.00. All KS p<0.02. Score: 8.0/10.
- **Idea 152 (Fiedler scaling)**: Exponents unstable for d>=4 (Fiedler~0). Only absolute value matters. Score: 5.0/10.
- **Idea 153 (Treewidth vs d)**: tw/N decreases with d (0.50->0.13) but exponents don't match (d-1)/d. Needs larger N. Score: 6.0/10.
- **Idea 154 (SVD compressibility vs d)**: alpha increases with d (0.78->1.03). Monotonic but NOT (d-1)/d. Score: 6.5/10.
- **Idea 155 (Geometric fingerprint)**: *** BEST RESULT *** Combined (Fiedler, tw/N, k/N) gives EXCELLENT separation. Cohen's d > 2.9 for ALL adjacent dimension pairs. Score: 8.5/10.
- **Idea 156 (Spectral gap vs d)**: Gap decreases with d. Less clean than Fiedler alone. Score: 6.0/10.
- **Idea 157 (Link density scaling)**: links/N ~ N^beta where beta matches (d-1)/d for d>=3 (within 0.06). Score: 7.0/10.
- **Idea 158 (Fiedler across BD)**: Fiedler jumps from 0.62 (beta=0) to ~1.5-2.2 (beta>0). Score: 6.5/10.
- **Idea 159 (Chain height scaling)**: h ~ N^{1/d} confirmed. d=3: 0.38 vs 0.33, d=4: 0.30 vs 0.25. R²>0.99. Score: 8.0/10.
- **Idea 160 (Antichain width scaling)**: *** BEST RESULT (tied) *** w ~ N^{(d-1)/d} confirmed. d=2: 0.51 vs 0.50, d=3: 0.66 vs 0.67. R²>0.99. Score: 8.5/10.
- **Headline**: Chain+antichain give DUAL exponents (1/d and (d-1)/d summing to 1). Geometric fingerprint (Idea 155) cleanly classifies d=2-5. These are the FIRST 8+ results in 160 ideas.
- **160 total ideas tested. 7.5 ceiling BROKEN. Best scores: 8.5/10 (Ideas 155 and 160).**

**[DONE] Cross-Dimensional Comparison — Ideas 171-180 (Exp65)**
- Strategy: extend observables across d=2,3,4,5,6 to find dimension-dependent signatures
- N=30 for d=2-5, N=20 for d=6, 5 trials each
- **Idea 171 (Fiedler value)**: NON-MONOTONIC. Collapses to ~0 for d≥4 (Hasse diagram disconnects). Score: 3/10.
- **Idea 172 (Treewidth proxy)**: CLEAN ENCODER. tw = 14→10.2→5.2→2.6→1.4 across d=2-6. Linear fit predicts d with max error 0.46. Score: 8/10.
- **Idea 173 (Compressibility)**: CLEAN ENCODER. eff_rank/N = 0.68→0.49→0.32→0.29→0.15 across d=2-6. Linear fit max error 0.50. Score: 8/10.
- **Idea 174 (Longest chain / N^{1/d})**: Decent. Ratio ≈ 0.84-1.11, roughly constant but noisy. Score: 6/10.
- **Idea 175 (Longest antichain / N^{(d-1)/d})**: NON-MONOTONIC (peaks then drops for d=6 at N=20). Ratio ≈ 1.4 fairly constant. Score: 3/10.
- **Idea 176 (Ordering fraction)**: Measured vs 1/d!: exact at d=2, DIVERGES for d≥3. Random d-orders ≠ sprinkled causets. Score: 6/10.
- **Idea 177 (Link fraction)**: DECENT ENCODER. 0.28→0.56→0.82→0.98→0.99. Saturates near 1 at high d. Score: 6/10.
- **Idea 178 (SJ c_eff)**: BEST ENCODER. c_eff = 0.98→0.93→0.63→0.49→0.24. Linear fit max error 0.39. Score: 8/10.
- **Idea 179 (Level spacing ratio)**: NON-MONOTONIC. ⟨r⟩ ≈ 0.50-0.63, GUE universality is dimension-independent. Score: 3/10.
- **Idea 180 (Interval entropy)**: DECENT ENCODER. H = 2.14→1.25→0.57→0.20→0.08. Linear fit max error 0.64. Score: 6/10.
- **Best encoders (score 8)**: treewidth proxy, compressibility (eff_rank/N), SJ c_eff. All three cleanly predict d to within 0.5 via simple linear formula.
- **Key insight**: Ordering fraction for random d-orders is NOT 1/d! (that's for sprinkled causets in causal diamonds).
- **Key negative**: Level spacing ratio and Fiedler value do NOT encode dimension. GUE is dimension-blind.
- 180 total ideas tested. Best single-observable dimension encoder: SJ c_eff (max error 0.39).

**[DONE] SYNTHESIS — Ideas 191-200 (Exp67)**
- Strategy: COMBINE multiple results into unified frameworks. Final 10 ideas.
- **Idea 191 (Multi-observable dimension)**: All three estimators (MM, chain, antichain) give consistent d for sprinkled causets. MM is best single estimator (bias <0.2 for d=2-4). Chain underestimates, antichain overestimates. Joint average slightly reduces bias. Score: 6.0/10.
- **Idea 192 (Info-theoretic phase classifier)**: PC1 of (H, compressibility, Fiedler, ordering fraction) captures 70% of variance. H and compressibility have sharpest transitions. No single quantity captures full phase diagram. Score: 5.5/10.
- **Idea 193 (RMT + graph synthesis)**: Fiedler-r correlation is WEAK (rho=-0.20, p=0.13). Fiedler-H is strong (rho=-0.53). Graph connectivity and spectral statistics probe INDEPENDENT aspects. Score: 6.0/10.
- **Idea 194 (Universality class)**: chi_max ~ N^1.46 (specific heat exponent). Consistent with second-order or strong first-order transition. N range (20-40) too small for definitive classification. Score: 5.5/10.
- **Idea 195 (Causet signature)**: BEST SINGLE DISCRIMINATOR is compressibility (Cohen's d = 9.46). Combined linear classifier achieves 100% accuracy on causets vs matched random DAGs. Score: 6.5/10.
- **Idea 196 (SJ vs geometry meta-analysis)**: Pure geometry ideas scored more consistently (6-7). SJ vacuum had higher ceiling (7.5) but deflated by null controls. Pure geometry is stronger foundation. Score: 5.0/10.
- **Idea 197 (FSS collapse)**: Best collapse at nu=1.17. Does not match standard universality classes (Ising nu=1, mean-field nu=0.5). Likely finite-size effects. Score: 5.5/10.
- **Idea 198 (Lee-Yang zeros)**: Exact Z(beta) for N=4: 9 action levels from S=-6 to S=10 across 576 2-order states. Closest zero to real axis at distance ~0.7 — approaching but not at real axis. Framework for future analysis. Score: 6.5/10.
- **Idea 199 (Joint dimension formula)**: MM alone is already near-optimal. Joint estimator has slightly larger bias at d>2 due to antichain overestimation. Score: 5.0/10.
- **Idea 200 (Meta-analysis)**: Best coherent paper: "BD Phase Transition: Exact Results, Critical Exponents, and Random Matrix Connection." Combines multiple themes. Honest ceiling: 7.5/10 — the 8+ barrier is structural (toy scale N=20-50).
- **200 total ideas tested. 7.5/10 ceiling confirmed as structural. 8 ideas scored >= 7.0 (4%). Search complete.**

**[DONE] WILD CARD ROUND 2 — Ideas 291-300 (Exp77/Round 20)**
- Strategy: 10 truly unconventional ideas — game theory, Kolmogorov complexity, quantum complexity, TDA, fractal dimension, Boolean networks, RG flow, Markov chains, max-flow, emergent topology
- **Idea 291 (Game Theory PoA)**: Price of Anarchy INCREASES with d (2.30→3.18). New dimension estimator from game theory. Score: 7.0/10.
- **Idea 292 (Kolmogorov Complexity)**: 2-order representation 2.5× more compact than matrix. Sprinkled 26% more compressible than random. Score: 6.5/10.
- **Idea 293 (SJ Quantum Complexity)**: **AREA LAW**: S ~ N^0.453 in d=2 (sqrt scaling). Score: 7.5/10.
- **Idea 294 (TDA Config Space)**: Effective dimension 1.0 vs theoretical 18. Highly constrained. Score: 6.0/10.
- **Idea 295 (Fractal Dimension)**: Decreases with d but may be spectral embedding artifact. Score: 5.5/10.
- **Idea 296 (Boolean Network)**: ALL causets at edge of chaos (Derrida ~0.01-0.02). Score: 7.0/10.
- **Idea 297 (RG Flow)**: Ordering fraction flows to 1.0 (total order). No nontrivial IR fixed point. Score: 6.5/10.
- **Idea 298 (Markov Chain)**: TOTAL FAILURE. Absorbs in 1 step. Score: 3.0/10.
- **Idea 299 (Max-Flow)**: Strongly dimension-dependent but doesn't match area law or correlate with SJ entropy. Score: 6.5/10.
- **Idea 300 (Emergent Topology)**: **TOPOLOGY DETECTED**: Diamond vs cylinder distinguished at p<0.001 via ordering fraction and boundary structure. Score: 7.5/10.
- **Top results**: Topology detection (7.5), SJ area law (7.5), edge of chaos (7.0), game theory PoA (7.0)
- **300 total ideas tested across 20 rounds. Best overall: 8.5/10 (geometric fingerprint, dual exponents).**

**[DONE] REVIEW PAPER SYNTHESIS — Ideas 311-320 (Exp79, in papers/exact-combinatorics/)**
- Strategy: COMBINE existing results into unified frameworks for a review/synthesis paper scoring 8+
- **Idea 311 (Dimension Estimator Table)**: 10 observables on d-orders d=2,3,4,5. Five observables achieve PERFECT |rho|=1.0 correlation with d: MM dim, ordering fraction, h/w ratio, path entropy, interval entropy. Score: 7.5/10.
- **Idea 312 (Phase Transition Table)**: 7 observables across beta=0 to 5*beta_c. S_BD/N has 1115% jump (100% monotonic). Link fraction 60% jump (partially monotonic). Fiedler 243% jump but noisy. Score: 7.0/10.
- **Idea 313 (Universal Scaling)**: links/N ~ ln(N), AC/sqrt(N) -> 2, chain/sqrt(N) -> 2, Fiedler*N -> constant, link_frac ~ N^{-0.63}. Score: 7.0/10.
- **Idea 314 (Master Formula d-Orders)**: P[k|m]=(m-k-1)/[m(m-1)] is SPECIFIC to d=2. For d=3, links are 1.78x more common than 2-order prediction; for d=4, 2.27x. Higher d concentrates interval distribution at k=0. Score: 7.5/10 — important negative result.
- **Idea 315 (Exact BD Action)**: E[S_BD] = N - 2*sum(N-d)/(d+1) + sum(N-d)(d-1)/[d(d+1)], verified exactly for N=4-7. E[S_BD]/N -> 0 (flat space). Score: 7.0/10.
- **Idea 316 (Vershik-Kerov d-Orders)**: AC ~ c_d * N^{(d-1)/d} confirmed. Fitted: c_2=1.43, c_3=1.22, c_4=1.10, c_5=1.24. Exponents match theory within ~0.1. Score: 7.5/10.
- **Idea 317 (Link + Fiedler -> Expansion)**: Average Hasse degree ~ 2*ln(N), lambda_2*N -> constant (~5-69, still growing). Cheeger lower bound ~ c/(2N) -> NOT expander. Score: 7.0/10.
- **Idea 318 (Null Model Ladder)**: 6 structures ranked by 7 observables. chain/N (Cohen's d=3.82) and S_BD/N (Cohen's d=3.84) best distinguish 2-order from sprinkled causet. 2-order closest to Sprinkle 2D (dist=0.68). Score: 7.5/10.
- **Idea 319 (Fisher Information)**: I(beta)=Var[S] peaks at beta~0.20 for N=4,5. I(beta)/N grows with N (MCMC N=30: peaks at beta_c=0.12 with I/N=26.7). Consistent with first-order transition. Score: 7.5/10.
- **Idea 320 (Paper Abstract)**: Complete abstract and 6-section outline written for "Exact Combinatorics of Causal Sets: Interval Statistics, Scaling Laws, and Dimension Detection in d-Orders". Score: 8.0/10 as paper concept.
- **320 total ideas tested. Synthesis paper concept scored 8.0/10 — the combination of exact formulas, systematic tables, and null controls produces the strongest single-paper contribution.**

**[DONE] KR Phase Deep Characterization — Ideas 341-350 (Exp82)**
- Strategy: deep-dive into the crystalline (KR) ordered phase of the BD transition
- **Idea 341 (Pure KR vs MCMC)**: Pure 3-layer KR has 3 layers, chain=3. MCMC KR at 5*beta_c has ~5-6 layers, chain=5-6. The MCMC ordered phase is a GENERALIZED layered "pancake stack", NOT purely 3-layered. Ordering fraction differs: 0.37 (pure) vs 0.51 (MCMC). Score: 7.5/10.
- **Idea 342 (KR Degeneracy)**: For N=50, there are 2^624 distinct 3-layer posets. Ground state is MASSIVELY degenerate — independent MCMC chains converge to similar S/N but different configs (CV=0.53). Score: 7.0/10.
- **Idea 343 (KR Entropy)**: S_raw/N ~ (a*b+b*c)*ln(2)/N grows quadratically. At N=50: S/N=8.65 (raw), 6.63 (net of automorphisms). This extensive entropy explains thermodynamic stability of ordered phase. Score: 6.5/10.
- **Idea 344 (SJ Vacuum on KR)**: Pure KR Wightman eigenvalues show PERFECT DOUBLET structure (every pair degenerate, ratio=1.000). <r>_PJ = 0.63 for pure KR, 0.52 for disordered, 0.47 for MCMC KR. MCMC KR has FEWER positive modes (16 vs 20). Score: 7.5/10.
- **Idea 345 (Cooling Dynamics)**: Gradual: layers ~6-7 at all beta/bc = 0-3, then max_width jumps at 5*bc (14.4 vs 11-12). Link fraction increases sharply at 5*bc (0.41 vs 0.25). Crystallization is abrupt in link fraction. Score: 6.5/10.
- **Idea 346 (Layer Count vs N)**: N=30: 4.3 layers; N=50: 5.0; N=100: 8.8; N=200: 11.4. Layer count GROWS with N — NOT fixed at 3. max_w/N = 0.45 (small N), decreasing to 0.18 (N=200). Score: 7.5/10 — disproves pure KR model.
- **Idea 347 (Fiedler Value)**: Pure KR Fiedler=6.97 (very high — dense bipartite links). MCMC KR=1.22, disordered=0.70. Ratio KR/disordered = 1.75x. KR Hasse is more algebraically connected. Score: 6.5/10.
- **Idea 348 (Manifold Properties)**: Pure KR: of=0.38, d_MM=1.42 (too low-dimensional). MCMC KR: of=0.49, d_MM=1.02. Sprinkled 2D: of=0.52, d_MM=0.95. KR ordering fraction is CLOSE to random 2-order — NOT manifold-like. Score: 6.0/10.
- **Idea 349 (Interval Distribution)**: ANALYTICALLY CONFIRMED. Links = 314 (predicted 312, all adjacent-layer). L0->L2 intervals follow Binomial(26, 0.25) with peak at k=6. P(L0->L2 link) = 0.75^26 = 5.6e-4. Score: 7.0/10.
- **Idea 350 (Predict <r>=0.12)**: Pure KR <r>=0.54, MCMC KR <r>=0.56. NEITHER reproduces <r>=0.12! The sub-Poisson statistic is specific to the TRANSITION region, not to the deep KR phase. Deep KR has GUE-like <r>~0.5. Score: 7.0/10 — important negative.
- **Key findings**: (1) MCMC KR phase is NOT the pure 3-layer KR poset — it has ~sqrt(N) layers. (2) Ground state is massively degenerate (2^624 at N=50). (3) The <r>=0.12 sub-Poisson statistic is a TRANSITION phenomenon, not a KR property. (4) SJ vacuum on pure KR shows perfect eigenvalue doublets from layer symmetry.
- **350 total ideas tested. Score: 7.0/10 overall.**

**[DONE] Strengthen Paper F — Ideas 411-420 (Exp90)**
- Strategy: Push Paper F (Hasse Geometry, 7.5/10) toward 8 with deeper spectral analysis, Cheeger constant, FSS, path entropy, universality class comparison.
- **Idea 413 (Fiedler eigenvector = spatial partition)**: BEST NEW RESULT. Fiedler vector correlates with spatial coordinate at r=0.55 (sprinkled), r=0.57 (2-orders). Lightcone correlation r=0.61. Binary partition agreement 73%. The spectral bisection IS a spatial bisection. Score: 7.5/10.
- **Idea 420 (2-orders vs sprinkled universality)**: All Hasse observables match within 3-14% between 2-orders and sprinkled causets. Same universality class confirmed. Score: 7.5/10.
- **Idea 414 (Cheeger constant)**: Causets h=0.56 vs DAGs h=0.13 at N=10 (4.3x). Direct expansion measurement. Score: 7.0/10.
- **Idea 411 (Fiedler lower bound)**: lambda_2 ~ 0.20*N^0.33 confirmed. Grows without bound. Score: 6.5/10.
- **Idea 416 (Fiedler jump FSS)**: ~50% jump from continuum to crystal. Score: 6.5/10.
- **Idea 417 (Path entropy)**: 16% drop across BD transition. Modest discriminator. Score: 6.5/10.
- **Idea 419 (Diameter)**: 21% diameter drop. Diameter ~ N^0.155, slower than sqrt(N). Score: 6.5/10.
- **Idea 412 (Spectral gap ratio)**: ~0.06, roughly constant with N. Score: 6.0/10.
- **Idea 415 (Link fraction FSS)**: Jump ~0.03 doesn't sharpen with N. Score: 6.0/10.
- **Idea 418 (Geometric health index)**: GHI rises 0.35→0.43. Moderate. Score: 6.0/10.
- **420 total ideas tested. Paper F stays at 7.5/10. Ideas 413 (spatial partition) and 420 (universality) are strongest additions.**

**[DONE] Continuum Limit Tests — Ideas 451-460 (Exp94)**
- Strategy: test whether causal sets reproduce KNOWN CONTINUUM RESULTS at accessible N=50-200. Every match strengthens the causal set programme.
- **Idea 451 (Weyl's Law)**: Eigenvalue counting N(λ) ~ λ^{d/2} for Hasse Laplacian. At N=50 inferred d=2.41, growing to d=3.17 at N=200. Does NOT converge to d=2 — eigenvalue counting function grows faster than Weyl's law predicts, likely due to graph irregularity. Score: 5.0/10.
- **Idea 452 (Minkowski box-counting dimension)**: d_box = 1.15 (N=100) to 1.58 (N=500). Systematically BELOW 2.0 — the diamond region's shape (not a square) biases box-counting. A calibration artifact, not a physics result. Score: 4.0/10.
- **Idea 453 (Hausdorff dimension)**: Ball-volume growth on Hasse graph distance gives d_H = 1.33 (N=50) → 1.91 (N=200). CONVERGING TOWARD 2.0 with increasing N. For d=3, gives 1.35-1.55 (also below target). The Hasse graph distance is a reasonable metric but finite-size effects are significant. Score: 7.0/10.
- **Idea 454 (Green's function)**: SJ Wightman vs continuum -(1/4π)ln|σ²|. Pearson r = 0.86-0.90, Spearman ρ = 0.91-0.93, IMPROVING with N. STRONG MATCH — the discrete SJ propagator faithfully approximates the 2D continuum Green's function. Score: 8.0/10.
- **Idea 455 (Propagator poles)**: Momentum-space |W̃(p)| ~ p^{-0.2 to -0.5}, far from the expected p^{-2}. R² values 0.02-0.17 (poor fits). The discrete FT on a causal set is ill-defined — no translation invariance means no clean momentum space. Score: 3.0/10.
- **Idea 456 (Spectral dimension)**: UNNORMALIZED Hasse Laplacian heat kernel: d_s = 1.49 (N=50) → 1.97 (N=200). CONVERGING BEAUTIFULLY TO 2.0. This is the key result — unlike the link-graph spectral dimension (which fails), the HASSE Laplacian heat kernel spectral dimension DOES recover d=2. Normalized Laplacian overshoots (2.59 at N=200). Score: 8.5/10 — BEST RESULT this round.
- **Idea 457 (Volume-distance scaling)**: V(r) ~ r^{2.14} at N=100, rising to r^{2.54} at N=200. Close to r^2 at small N but drifts upward. Chain distance version produced too few valid fits. Score: 6.0/10.
- **Idea 458 (Euler characteristic)**: χ = V - E + T. Triangle-free Hasse graph means T=0, so χ = V - E. At N=200: χ ≈ -576, far from 0. The clique complex of the Hasse diagram does NOT recover topology — the Hasse graph is too sparse (tree-like) for simplicial methods. Score: 3.5/10.
- **Idea 459 (Geodesic deviation)**: Nearby chain divergence exponent = 0.2-0.5, well below the expected 1.0 for flat 2D. Chains are too short and discrete for Jacobi field analysis at N~200. Score: 4.0/10.
- **Idea 460 (BD action vs curvature)**: S_BD/N ≈ -3.6 for ALL curvatures R = -2 to +4. No statistically significant dependence on R (Pearson r = 0.42, p = 0.30). The conformal-metric sprinkling approach may not produce enough curvature contrast at this N. Score: 4.5/10.
- **Top results**: Idea 456 (spectral dimension d_s → 2.0, 8.5/10), Idea 454 (Green's function r=0.90, 8.0/10), Idea 453 (Hausdorff dimension → 2.0, 7.0/10).
- **Key finding**: The UNNORMALIZED Hasse Laplacian heat kernel spectral dimension CONVERGES TO d=2 as N grows — this is the first spectral dimension estimator that works on causal sets. The SJ Wightman function strongly correlates with the continuum Green's function (r=0.90). These two results are the strongest continuum-limit confirmations in the project.
- **460 total ideas tested. New best: 8.5/10 (Hasse Laplacian spectral dimension → 2).**

**[DONE] Round 21 — Fiedler Value Analytics, Ideas 331-340 (Exp81)**
- Strategy: Push Paper F's main result (Fiedler value, 7.5/10) toward 8+ with ANALYTIC understanding
- **Idea 331 (Path graph)**: VERIFIED. λ₂(chain) = 2(1-cos(π/N)) ~ π²/N². Machine-precision match. Score: 6.0/10 (known result, but establishes baseline).
- **Idea 332 (Antichain)**: VERIFIED. λ₂ = 0 (disconnected graph). Score: 5.0/10.
- **Idea 333 (Interpolation chain→random)**: λ₂ rises rapidly from π²/N² to ~0.7 after ~N/2 transpositions, then saturates. Adding spacelike structure drastically improves connectivity. Score: 7.0/10.
- **Idea 334 (λ₂ vs L/N)**: Pearson r=0.46, Spearman ρ=0.53. Power law λ₂ ~ (L/N)^2.2. More links → higher algebraic connectivity. Score: 6.5/10.
- **Idea 335 (λ₂ vs ordering fraction)**: r=0.27, partial correlation controlling for L/N is r=0.26. λ₂ correlates with f INDEPENDENTLY of link count. Score: 6.0/10.
- **Idea 336 (Distribution)**: NOT Gaussian (Shapiro-Wilk p=1.4e-12). Left-skewed (skew=-0.64) with excess kurtosis (1.45). Spike at λ₂≈0 from disconnected Hasse diagrams. Score: 6.5/10.
- **Idea 337 (Scaling)**: <λ₂> ~ N^{0.318 ± 0.012}, R²=0.993. Consistent with previously reported N^0.34. Score: 7.5/10 — the key analytic result.
- **Idea 338 (Expander check)**: Cheeger inequality h ≥ λ₂/(2d_max) verified for ALL tested cases. But d_max ~ N^0.45 while λ₂ ~ N^0.32, so h_lower ~ N^{-0.13} → NOT an expander family. The Cheeger constant shrinks with N. Score: 7.0/10 — important negative result.
- **Idea 339 (2-order vs sprinkled)**: 2-order λ₂ ~ N^0.25, sprinkled λ₂ ~ N^0.35. Consistent within 2σ. 2-orders faithfully reproduce sprinkled Hasse connectivity. Score: 7.0/10.
- **Idea 340 (SJ vacuum correlation)**: λ₂ vs c_eff: r=0.31 (p=0.002). SIGNIFICANT — Hasse connectivity encodes information about the quantum field vacuum. λ₂ vs spectral gap: r=-0.13 (not significant). Score: 7.5/10.
- **Key analytic results**: (1) λ₂ bounded between 0 (antichain) and ~π²/N² (chain), with random 2-orders at N^0.32. (2) NOT an expander family — d_max grows faster than λ₂. (3) 2-orders match sprinkled causets. (4) λ₂ correlates with SJ vacuum c_eff.
- **340 total ideas tested. Paper F strengthened with analytic bounds and scaling verification.**

**[DONE] Round 18 — Computational/Algorithmic Novelty, Ideas 271-280 (Exp75)**
- Strategy: bring modern CS/ML techniques to causal set theory — sparse eigensolvers, randomized SVD, ML classifiers, parallel tempering, spectral clustering, community detection, PageRank, graph wavelets, persistent homology, tensor decomposition
- **Idea 271 (Sparse SJ vacuum)**: Pushed SJ eigenvalues to N=1000 using scipy.sparse.linalg.eigsh (~108s). Level spacing ratio <r>=0.53-0.62 across N=200-1000, consistent with GUE. Enables large-N SJ studies. Score: 7.0/10.
- **Idea 272 (Randomized SVD)**: 90% energy captured by k=8 modes (d=2), k=16 (d=3), k=12 (d=4). Effective rank encodes dimension. Sparse SVD 90x faster than dense. Score: 7.0/10.
- **Idea 273 (RF dimension classifier)**: 99.3% cross-validated accuracy classifying d=2,3,4 from graph features. Top features: ordering fraction, mean interval, std interval, longest chain. A GNN would learn the same. Score: 7.5/10.
- **Idea 274 (Parallel tempering)**: Replica exchange MCMC at N=80,120. Swap acceptance 3.5-12.7%. Specific heat C_v grows with beta^2*Var(S). Ordering fraction varies across replicas. Score: 6.5/10.
- **Idea 275 (Spectral clustering)**: Clusters RECOVER spatial regions. Variance explained: time 64%, space 51% in d=2. Spectral gap larger for sprinkled causets than null random DAGs. Score: 7.5/10.
- **Idea 276 (Community detection)**: Greedy modularity finds 5-6 communities with Q~0.38. Communities are spatially coherent (space variance explained 69%). Modularity decreases with d. Score: 6.5/10.
- **Idea 277 (PageRank)**: STRONG correlation with time coordinate: Spearman rho=0.90 (d=2), 0.89 (d=3). High-PR elements are at late times. PR provides purely causal "importance" that recovers temporal position. Score: 7.5/10.
- **Idea 278 (Graph wavelets)**: Time signal energy ratio 126x that of random at scale 1.83. Time is a large-scale feature of causal structure. Multi-scale decomposition works on Hasse Laplacian. Score: 6.5/10.
- **Idea 279 (Persistent homology)**: Chain-distance filtration: Hasse diagram already connected at r=1 (dense links). b1=167-194 at r=1 (many loops), drops to 0 at r=2. Limited discriminating power at these sizes. Score: 5.5/10.
- **Idea 280 (Tensor decomposition)**: CP factors correlate with time (sink factor rho=0.81). Rank-10 captures 58% of tensor. Tucker core shows hierarchical structure. Tensor density strongly dimension-dependent. Score: 6.5/10.
- **Top results**: PageRank (7.5), spectral clustering (7.5), RF classifier (7.5), sparse SJ (7.0), randomized SVD (7.0)
- **Key finding**: Modern graph algorithms (PageRank, spectral clustering, wavelets) RECOVER spacetime geometry from purely causal data, computationally confirming "order + number = geometry."
- **280 total ideas tested. Best overall: 8.5/10 (geometric fingerprint, dual exponents from Exp63).**

**[DONE] Round 14 — Information-Theoretic Properties, Ideas 231-240 (Exp71)**
- Strategy: information-theoretic analysis of causal matrices — algorithmic complexity, conditional entropy, mutual information, source coding, rate-distortion, Fisher information
- **Idea 231 (Lempel-Ziv complexity)**: STRONG DISCRIMINATOR. Cohen's d = 4.97 (causet vs DAG). CLEANLY ENCODES DIMENSION: LZ_norm decreases monotonically from d=2 (1.61) to d=4 (0.98). Score: 7.5/10.
- **Idea 232 (Conditional row entropy)**: H(j|i) is LOWER for causally related pairs (0.22) than unrelated (0.35) in 2D causets. Gap shrinks with dimension. Interesting but small N. Score: 6.0/10.
- **Idea 233 (Past-future MI)**: EXCELLENT DISCRIMINATOR. Cohen's d = 7.25 (causet vs DAG). Strong dimension sensitivity (3.3 sigma between 2D and 4D). Causets have MUCH LESS past-future MI than random DAGs — geometry constrains information flow. Score: 8.0/10.
- **Idea 234 (Info-theoretic dimension)**: Row-sum entropy MONOTONICALLY DECREASES with d (2.83→0.91 for d=2→5). Cleaner than binary entropy. Score: 6.5/10.
- **Idea 235 (Source coding bound)**: BEST DISCRIMINATOR THIS ROUND. Cohen's d = 8.54 for bits/relation (causet vs DAG). Causets need ~3 bits/relation, DAGs ~1.4. Geometric causets pack MORE info per relation. Score: 8.0/10.
- **Idea 236 (Rate-distortion)**: Effective rank/N cleanly separates types (0.70 for 2D causet, 0.77 for DAG, 0.34 for 4D causet). Rate for <5% distortion: causets ≈ DAGs at d=2. Score: 6.5/10.
- **Idea 237 (Fisher information)**: Var(S) peaks at 5*beta_c, NOT at beta_c. Fisher info increases monotonically — no sharp peak at transition for N=40. Likely needs larger N. Score: 5.5/10.
- **Idea 238 (Randomness deficiency)**: WEAK. Cohen's d = 0.52. All matrices (causets and DAGs) compress better than iid at matched density. No clean separation. Score: 4.5/10.
- **Idea 239 (Layer screening CMI)**: TOTAL NULL. Screening = 1.000 for ALL types — layers perfectly screen past from future in both causets and DAGs at this N. Cohen's d = 0.00. Score: 3.0/10.
- **Idea 240 (Entropy production rate)**: sigma(dH) DECREASES monotonically with beta (0.060 → 0.019), autocorrelation increases. Detects ordering but not a sharp transition. Score: 5.5/10.
- **Top results**: Idea 233 (past-future MI, d=7.25, 8.0/10), Idea 235 (source coding, d=8.54, 8.0/10), Idea 231 (LZ complexity, d=4.97, 7.5/10)
- **Key insight**: Geometric causets carry MORE information per relation than random DAGs — they are informationally richer, not simpler. This is consistent with Exp62 Idea 150 (information bottleneck).
- **240 total ideas tested. Two new 8.0/10 results (past-future MI and source coding bound).**

**[DONE] Analytic Breakthroughs — Ideas 471-480 (Exp96)**
- Strategy: Attempt genuine 8+ results via PROOFS with numerical verification
- **Idea 474 (Link fraction exact formula)**: **HEADLINE RESULT (8.0/10)**. PROVED: link_frac = 4((N+1)H_N - 2N)/(N(N-1)) ~ 4ln(N)/N. Measured/formula ratio = 1.00 at N=320. The "N^{-0.72} power law" is NOT a power law — it's 4ln(N)/N with running effective exponent 1-1/ln(N).
- **Idea 471 (Hasse connectivity)**: PROVED (7.5/10). E[deg] ~ 2ln(N), min degree → ∞. P(connected) = 1.000 for N ≥ 160.
- **Idea 477 (Tracy-Widom rate)**: VERIFIED (7.5/10). (LA-2√N)/N^{1/6} → TW₂ with mean → -1.77, var → 0.81.
- **Idea 473 (Fiedler → ∞)**: PROVED conditionally via Cheeger (7.0/10). λ₂ ~ N^{0.28}.
- **Idea 476 (Chain-AC independence)**: CONFIRMED (7.0/10). Known result (Baik-Rains 2001). Correlation → 0.
- **Idea 478 (Interval unimodality)**: PROVED conditionally (7.0/10). E[N_k] strictly decreasing for all N tested.
- **Idea 479 (Spectral gap)**: Gap ~ N^{-1.98} (6.5/10). Much faster than 1/N.
- **Idea 475 (E[S_BD])**: Formula derived but E[N_k] analytic formula off by factor ~2 (6.5/10).
- **Idea 472 (Eigenvalue density)**: Kurtosis ~ N (6.0/10). Exact density open.
- **Idea 480 (Partition zeros)**: Z(β) > 0 for real β (trivially positive) (5.0/10).
- **480 total ideas tested. New exact closed-form result at 8.0/10.**

**[PARTIAL] Connection to experimental data (LIV, gravitational decoherence, primordial spectrum)**
- Lorentz invariance violation (Exp51): Extracted SJ dispersion relation at N=50-400, compared with LHAASO constraints
  - Momentum-space Fourier approach too noisy at N≤200 (alpha_1 consistent with zero but huge error bars)
  - Eigenvalue spectrum: SJ eigenvalues ~2.7× continuum at low modes, converging at high modes; beta_1~3.5 but this is overall normalization, not LIV
  - Position-space rapidity test: |corr(W, rapidity)| = 0.039 ± 0.022 → SJ vacuum approximately Lorentz-invariant, NOT distinguishable from null (0.37 sigma)
  - Honest score: 3/10 — proof of concept only, N too small for definitive conclusions
  - Key insight: sprinkling preserves Lorentz symmetry, so SJ vacuum on sprinklings SHOULD be approximately LI
- [ ] Gravitational decoherence: compute decoherence rate from SJ vacuum, compare Diosi-Penrose prediction
- [ ] Primordial spectrum: SJ vacuum on de Sitter causal set → P(k) → compare Planck n_s=0.965

### HIGH PRIORITY

**[ ] Interactive web app (React + Vite + R3F/Plotly)**
- Replace `report.html` with a proper research dashboard like our other projects
- Features: paper summaries with expandable detail, interactive charts (phase transition, ER=EPR scatter, quantum chaos bars, CDT scaling), experiment browser, 3D causal set visualization, guide tab
- Stack: React 19 + TypeScript + Vite + Tailwind + Plotly/Chart.js (or R3F for 3D causal set viz)
- Deploy on Vercel like other projects
- `report.html` is now deprecated — do not update it
- Reference: `somewhat_laymans_terms.html` for content/narrative structure

### Tier 1: Ready to Pursue

**[ ] SJ on CDT paper (Paper E, potential 8/10 after exp91)**
- Data exists (exp44, exp85, exp91): CDT c→1, causets c→∞, mechanism fully explained
- **Exp91 breakthrough**: Kronecker product theorem proves n_pos(iΔ) = n_pos(A_T) exactly — positive modes depend ONLY on T (time slices), not N. CDT: n_pos ~ T/2 ~ O(√N). Causets: n_pos ~ N/2. This is the analytic mechanism for c≈1 vs c→∞.
- Additional exp91 findings: 1% rewiring destroys c≈1 (structural fragility), 5% within-slice disorder doubles c_eff (rank jumps from 8 to 54), CDT Hasse is triangle-free, c_eff stable ~1.0 at N=40-120
- First cross-approach SJ vacuum comparison with analytic explanation
- Needs paper writing + peer review

**[ ] Reconstruct metric from W (Ideas V3 #8, potential 9/10)**
- Can we recover the causal set (or embedding coordinates) from the Wightman function alone?
- If spacetime can be reconstructed from entanglement data, that's "spacetime from entanglement"
- High risk, high reward

**[ ] Massive scalar SJ vacuum (Ideas V3 #11)**
- Natural extension: G_R = (1/2)C(I + m²/(2ρ)C)⁻¹
- How does mass affect entanglement, monogamy, ER=EPR?

### Tier 2: Would Strengthen Existing Papers

**[ ] 4D parallel tempering for interval entropy**
- 4D three-phase result (Paper A) is at N=30-70 without parallel tempering
- Acceptance drops to 17% at high β — thermalization concern
- Parallel tempering on 4-orders would solidify the result

**[ ] More β values for large-N 2D FSS**
- exp32 shows χ_max at the edge of the β ladder for N=150, 200
- Denser β coverage would give cleaner FSS exponents

### Tier 3: Exploratory

**[ ] Unruh effect on causal sets (Ideas V3 #4)**
**[ ] SYK model comparison (Ideas V3 #7)**
**[ ] Quantum extremal surfaces / islands (Ideas V3 #5)**
**[DONE] Lorentz invariance violation from SJ propagator (Ideas V3 #1, Exp51)**
- Result: SJ vacuum approximately Lorentz-invariant at N≤400, consistent with theoretical expectation
- Next: needs N~5000+ with sparse methods to probe discreteness-scale corrections

---

## Lessons Learned

See `LESSONS_LEARNED.md` for the full list. Key rules:

1. **Null model first** — random graph control before any claim
2. **Calibrate before exploring** — reproduce a published number first
3. **Score honestly (1-10)** — use the score to pivot or deepen
4. **Peer review before "done"** — simulate expert reviewers, run missing controls
5. **Follow the surprise** — unexpected results signal real physics
6. **Theorems > data** — even trivial proofs elevate a paper
7. **One strong claim per paper** — focus beats breadth

---

## Running Experiments

```bash
# Run any experiment
/usr/bin/python3 experiments/expNN_name.py

# Compile a paper
cd papers/<paper-dir> && tectonic <paper>.tex

# Open reports
open report.html
open somewhat_laymans_terms.html
open papers/SUMMARY.md
```

---

## References

Core papers this project builds on:
- Bombelli, Lee, Meyer, Sorkin (1987) — causal sets
- Rideout & Sorkin (2000) — CSG dynamics
- Ahmed, Dodelson, Greene, Sorkin (2004) — everpresent Λ
- Surya (2019) — causal set review
- Benincasa & Dowker (2010) — causal set action
- Surya (2012), Glaser et al. (2018) — BD phase transition
- Johnston (2009), Sorkin (2012) — SJ vacuum
- Ambjørn, Jurkiewicz, Loll (2005) — CDT spectral dimension
- Carlip (2017) — dimensional reduction review
- Maldacena & Susskind (2013) — ER=EPR
- Hayden, Headrick, Maloney (2013) — holographic monogamy
- Calabrese & Cardy (2004) — entanglement entropy in CFT
- Bohigas, Giannoni, Schmit (1984) — quantum chaos and RMT

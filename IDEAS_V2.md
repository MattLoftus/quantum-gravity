# Ideas V2: Fresh Directions

Generated after completing 5 papers (A: 7, B1: 7, B2: 5, B3: 7, B5: 7.5), 42 experiments, and extensive exploration of causal sets, spectral dimension, holographic codes, and cosmology.

Criteria for each idea:
- **Score ceiling**: best-case paper quality (1-10)
- **Feasibility**: can we do it with our current tools? (High/Medium/Low)
- **Novelty**: has anyone done this before? (High = nobody, Medium = related work exists, Low = done)
- **Effort**: time to first result (hours)
- **Risk**: probability it produces nothing useful

---

## 1. Causal set entanglement thermodynamics

**Idea:** Compute the SJ entanglement entropy as a function of "temperature" (β) and extract a thermodynamic equation of state. If S(β) satisfies a first-law relation dS = dE/T with E = BD action and T = 1/β, the causal set has emergent thermodynamics. This would connect to Jacobson's derivation of Einstein equations from thermodynamics.

**Score ceiling:** 9 — emergent thermodynamics from discrete QG is a major result
**Feasibility:** High — we have S(β) data from exp35 and action data from exp29
**Novelty:** High — nobody has checked the first law on causal set entanglement
**Effort:** 2 hours (reanalyze existing data, compute dS/dβ and compare with d<S_action>/dβ)
**Risk:** Medium — the first law may not hold because β is a coupling, not a physical temperature

---

## 2. Entanglement entropy of sprinkled causets in CURVED spacetimes

**Idea:** Sprinkle into 2D spacetimes with varying curvature (de Sitter, anti-de Sitter, black hole) and measure how the SJ entanglement entropy depends on curvature. In the continuum, entanglement entropy gets curvature corrections: S = (c/3)ln(l/ε) + (curvature terms). Can we see these corrections on the causal set?

**Score ceiling:** 8 — detecting curvature through entanglement on a discrete structure
**Feasibility:** Medium — we have de Sitter sprinkling code, need AdS sprinkling
**Novelty:** High — SJ entropy on curved causal sets hasn't been studied systematically
**Effort:** 3-4 hours
**Risk:** Medium — curvature corrections may be too small at accessible N

---

## 3. Information scrambling time on causal sets

**Idea:** How fast does information spread through a causal set? Define a "scrambling time" t* as the time (in MCMC steps or proper time) for a local perturbation to affect the SJ correlations across the entire system. In holographic systems, t* ~ (1/T) ln(S), where S is the entropy (the fast scrambling conjecture). Does the causal set continuum phase scramble fast?

**Score ceiling:** 8-9 — fast scrambling is a hallmark of black holes and holographic systems
**Feasibility:** Medium — need to implement local perturbations and track correlation spreading
**Novelty:** High — scrambling on causal sets is completely unexplored
**Effort:** 4-5 hours
**Risk:** High — defining "scrambling" on a non-dynamical causal set is conceptually tricky

---

## 4. Causal set renormalization group flow

**Idea:** Define a coarse-graining operation on causal sets (randomly remove a fraction of elements, or merge nearby elements) and track how observables change under coarse-graining. This defines a renormalization group (RG) flow. Does interval entropy have a fixed point? Does the SJ entanglement entropy flow toward a CFT value? Does the BD action flow toward the continuum Einstein-Hilbert action?

**Score ceiling:** 9 — RG flow on discrete QG structures is a fundamental open problem
**Feasibility:** Medium — coarse-graining is straightforward (random decimation), but interpreting the flow requires care
**Novelty:** High — RG on causal sets is barely explored (Eichhorn has some work on AS + causal sets but not this approach)
**Effort:** 3-4 hours
**Risk:** Medium — the coarse-graining might not produce a sensible flow

---

## 5. Quantum error correction properties of the BD continuum phase

**Idea:** Treat the SJ Wightman function W as an encoding map from "bulk" (interior elements) to "boundary" (maximal/minimal elements). Test the Almheiri-Dong-Harlow conditions: can a bulk operator be reconstructed from a boundary subregion? Is the encoding an approximate quantum error correcting code? Does the continuum phase have better error correction properties than the crystalline phase?

**Score ceiling:** 9-10 — if the continuum phase IS a QEC code, it directly connects causal sets to the holographic program
**Feasibility:** Low-Medium — defining bulk/boundary on a causal set is ambiguous
**Novelty:** High — nobody has tested QEC properties on causal sets
**Effort:** 5-6 hours
**Risk:** High — the bulk/boundary decomposition may not be well-defined

---

## 6. Causal set analogue of the c-theorem

**Idea:** Zamolodchikov's c-theorem says that the central charge c decreases under RG flow in 2D. We measured c ≈ 3 for the SJ vacuum. If we coarse-grain the causal set (remove elements) and remeasure c, does it decrease? If c decreases monotonically under coarse-graining, the causal set satisfies a discrete analogue of the c-theorem.

**Score ceiling:** 9 — a discrete c-theorem from quantum gravity would be a major result
**Feasibility:** Medium — requires implementing coarse-graining + remeasuring SJ entropy
**Novelty:** High — no discrete c-theorem exists for causal sets
**Effort:** 3 hours (combines ideas 4 and the SJ infrastructure)
**Risk:** Medium — c might not change monotonically (our c=3 is already a normalization artifact)

---

## 7. Mutual information structure of causal diamonds at different scales

**Idea:** For a sprinkled causal set, consider nested causal diamonds of increasing size. Compute the SJ mutual information between the interior and exterior of each diamond. In AdS/CFT, this gives the Ryu-Takayanagi surface area. Does the causal set mutual information scale as the boundary area of the diamond?

**Score ceiling:** 8 — direct test of RT on causal sets with proper quantum observable
**Feasibility:** High — nested diamond partition is straightforward (exp39 already does this)
**Novelty:** Medium-High — extends exp39 with mutual information instead of just entropy
**Effort:** 2 hours
**Risk:** Low-Medium — the 2D area is trivial (a point), so the test is only meaningful in 4D

---

## 8. Phase transition in the SJ vacuum itself

**Idea:** As the causal set transitions from continuum to crystalline, does the SJ vacuum undergo its own phase transition? Specifically: is there a discontinuity in the Wightman function W, or in its eigenvalue distribution, at β_c? A vacuum phase transition would mean the quantum state on the causal set is fundamentally different in the two phases — not just the geometry.

**Score ceiling:** 8 — vacuum phase transitions are a hot topic (false vacuum decay, etc.)
**Feasibility:** High — we already compute W at each β, just need to track its eigenvalue distribution
**Feasibility:** 2 hours
**Risk:** Low — we'll definitely see SOMETHING change; the question is whether it's sharp

---

## 9. Tensor network representation of the SJ vacuum

**Idea:** Can the SJ Wightman function W be efficiently represented as a tensor network (MPS, MERA, etc.)? If yes: in the continuum phase, what is the bond dimension? Does it scale as a MERA (consistent with AdS/CFT)? In the crystalline phase, is the bond dimension smaller (less entangled = simpler tensor network)?

**Score ceiling:** 8-9 — connecting causal set QG to tensor networks directly
**Feasibility:** Medium — need to implement MPS/MERA decomposition of W
**Novelty:** High — nobody has done tensor network decomposition of the SJ vacuum
**Effort:** 4-5 hours
**Risk:** Medium — W may not have a natural tensor network structure

---

## 10. Predicting causal set dimension from entanglement scaling

**Idea:** We showed S ~ ln(N) in 2D (c/3 ≈ 1.0). In d dimensions, entanglement entropy scales as S ~ N^{(d-2)/d} for d > 2, or S ~ ln(N) for d = 2. If we measure S(N) on d-orders for d = 2, 3, 4, 5, does the scaling exponent correctly recover the dimension? This would give an "entanglement dimension estimator" — using quantum correlations instead of classical observables to measure dimension.

**Score ceiling:** 8 — a quantum dimension estimator is both novel and useful
**Feasibility:** High — we have d-orders and SJ vacuum code for arbitrary d
**Effort:** 3 hours (run at d=2,3,4,5 with N=20-60)
**Risk:** Medium — finite-size effects may prevent distinguishing the exponents

---

## 11. Causal set Bekenstein bound

**Idea:** The Bekenstein bound says the entropy of a region is bounded by S ≤ 2πER (in natural units) where E is the energy and R is the size. On a causal set, define E from the BD action and R from the longest chain. Does the SJ entanglement entropy satisfy S_SJ ≤ f(S_BD, chain_length) for some function f? If yes, the causal set satisfies a discrete Bekenstein bound.

**Score ceiling:** 8 — a discrete Bekenstein bound from first principles
**Feasibility:** High — we already have all quantities computed
**Effort:** 1-2 hours (scatter plot of S_SJ vs S_BD × chain_length across β)
**Risk:** Medium — the bound may not have a clean form

---

## 12. Apply our tools to causal DYNAMICAL triangulations

**Idea:** We built both CDT and causal set simulators. Compute the SJ vacuum entanglement entropy on CDT configurations (treating the triangulation as a causal set). Does CDT satisfy monogamy? Does CDT show the same CFT scaling as causal sets? This would be the first SJ vacuum computation on CDT — bridging two approaches through a shared quantum observable.

**Score ceiling:** 8-9 — nobody has computed the SJ vacuum on CDT configurations
**Feasibility:** Medium — need to extract a causal order from the CDT triangulation
**Novelty:** High — completely new
**Effort:** 4-5 hours
**Risk:** Medium — CDT has a regular lattice structure that may trivialize the SJ construction

---

## 13. Everpresent Lambda from the SJ vacuum energy

**Idea:** The SJ vacuum has a vacuum energy (expectation value of the stress tensor). On a causal set, this vacuum energy IS a cosmological constant. Compute <T_μν> for the SJ vacuum on flat causal sets. Does it naturally give Λ ~ 1/√N (the Sorkin prediction)? This would derive the everpresent Lambda from quantum field theory on causal sets, rather than from the action fluctuation argument.

**Score ceiling:** 9 — deriving the cosmological constant from QFT on causal sets
**Feasibility:** Low-Medium — computing <T_μν> on a causal set requires the stress tensor operator, which is subtle
**Novelty:** High — this connection hasn't been made computationally
**Effort:** 5-6 hours
**Risk:** High — the stress tensor on a causal set is not well-defined

---

## 14. Machine learning the BD phase boundary

**Idea:** Train a neural network to classify causal sets as continuum vs crystalline using the raw causal matrix C as input (not hand-crafted features like interval entropy). The network would learn its own features. Then analyze what the network learned — does it rediscover interval entropy? Or does it find something better?

**Score ceiling:** 7-8 — if the network finds a better observable than interval entropy, that's a discovery
**Feasibility:** High — standard PyTorch, we have labeled data
**Novelty:** Medium — ML for phase classification has been done in lattice QCD and condensed matter, but not for causal set QG
**Effort:** 3-4 hours
**Risk:** Low — the network will certainly learn SOMETHING; the question is whether it's interpretable

---

## 15. Causal set analogue of the ER=EPR conjecture

**Idea:** The ER=EPR conjecture (Maldacena-Susskind) says entanglement between two systems is equivalent to a wormhole connecting them. On a causal set: take two causally disconnected regions A and B. Measure their SJ mutual information I(A:B). If I(A:B) > 0, there are "quantum correlations" despite no causal connection. Now check: is there an "emergent causal connection" between A and B through the interval structure (elements that are related to both A and B elements)? If the strength of the emergent connection correlates with the mutual information, that's a discrete ER=EPR.

**Score ceiling:** 9-10 — a discrete verification of ER=EPR would be a landmark
**Feasibility:** Medium — defining "emergent causal connection" is the hard part
**Novelty:** High — nobody has tested ER=EPR on causal sets
**Effort:** 4-5 hours
**Risk:** High — the conjecture may not have a clean discrete analogue

---

## 16. Universal properties of the BD transition across dimensions

**Idea:** We have transition data in 2D and 4D. Add 3D (3-orders) and 5D (5-orders). Map how the critical exponents, the entropy jump ΔH, and the phase structure change with dimension. Are there universal features (same critical exponents in all dimensions)? Or does each dimension have a qualitatively different transition (as suggested by the 4D three-phase result)?

**Score ceiling:** 8 — universality (or lack thereof) across dimensions is a fundamental question
**Feasibility:** High — d-orders work for any d, computation time is manageable up to d=5
**Effort:** 4-5 hours
**Risk:** Low — we'll definitely get data; the question is whether it's interesting

---

## 17. Spectral gap and mass gap in the BD continuum phase

**Idea:** The smallest nonzero eigenvalue of the Pauli-Jordan operator (iΔ) defines a "spectral gap" — the minimum energy of excitations on the causal set. In the continuum phase, does this gap scale as 1/N (gapless, like a CFT) or as a constant (gapped, like a massive theory)? If the continuum phase is gapless, it's further evidence for CFT-like behavior. If the crystalline phase is gapped, the transition is also a gap-closing transition.

**Score ceiling:** 7-8 — the spectral gap is a fundamental characterization of the quantum theory
**Feasibility:** High — we already eigendecompose iΔ, just need to track the smallest eigenvalue
**Effort:** 1 hour
**Risk:** Low — pure analysis of existing data

---

## Ranking by Expected Value (Score × P(success))

| Rank | Idea | Score | P(success) | EV | Effort |
|------|------|-------|------------|-----|--------|
| 1 | #17 Spectral gap | 7.5 | 90% | 6.8 | 1h |
| 2 | #1 Entanglement thermodynamics | 9 | 60% | 5.4 | 2h |
| 3 | #8 Vacuum phase transition | 8 | 70% | 5.6 | 2h |
| 4 | #11 Bekenstein bound | 8 | 60% | 4.8 | 2h |
| 5 | #10 Entanglement dimension | 8 | 60% | 4.8 | 3h |
| 6 | #16 Universal properties across d | 8 | 60% | 4.8 | 4h |
| 7 | #4 RG flow | 9 | 50% | 4.5 | 3h |
| 8 | #14 ML phase boundary | 7.5 | 70% | 5.3 | 3h |
| 9 | #6 Discrete c-theorem | 9 | 45% | 4.1 | 3h |
| 10 | #2 Curved spacetime entanglement | 8 | 50% | 4.0 | 4h |
| 11 | #7 Nested diamond MI | 8 | 50% | 4.0 | 2h |
| 12 | #12 SJ vacuum on CDT | 8.5 | 45% | 3.8 | 5h |
| 13 | #3 Scrambling time | 8.5 | 40% | 3.4 | 5h |
| 14 | #9 Tensor network of SJ | 8.5 | 35% | 3.0 | 5h |
| 15 | #5 QEC properties | 9.5 | 30% | 2.9 | 6h |
| 16 | #15 Discrete ER=EPR | 9.5 | 25% | 2.4 | 5h |
| 17 | #13 Lambda from SJ vacuum energy | 9 | 25% | 2.3 | 6h |

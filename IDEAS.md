# Ideas for Future Directions

Generated after completing 4 papers (scores 5, 7, 5, 8) across causal sets, spectral dimension, cosmology, and holographic entanglement.

Guiding principle: maximize (score potential × feasibility), with a bias toward high-ceiling ideas even at lower feasibility.

---

## Tier S: Potential 9-10 (Would be genuinely important if they work)

### S1. Entanglement entropy produces the Bekenstein-Hawking formula on causal sets

**Idea:** Construct a causal set black hole (a sprinkling into a Schwarzschild causal diamond), compute the SJ entanglement entropy of the exterior, and check if S = A/(4G) with the correct coefficient.

**Why this is huge:** The Bekenstein-Hawking formula is the single most important result in quantum gravity. Bombelli, Koul, Lee, and Sorkin (1986) originally proposed that BH entropy IS entanglement entropy — but they couldn't compute the coefficient because it diverges in the continuum. On a causal set, the discreteness provides a natural UV cutoff. If the SJ entropy on a causal set Schwarzschild sprinkling gives S = A/(4l_P²) with the correct 1/4 coefficient, that would be a landmark result.

**Feasibility:** Medium. We can sprinkle into a 2D Schwarzschild spacetime (which is conformally flat, so the causal structure is tractable). The SJ vacuum on curved backgrounds has been studied (Afshordi, Aslanbeigi, Sorkin 2012). The main challenge: properly defining the "horizon" and "exterior" in the discrete setting. System sizes N=100-500 should be sufficient.

**Score if it works:** 9-10. This would directly connect causal set discreteness to black hole thermodynamics.

**Score if it partially works (wrong coefficient):** 7. Still interesting — the coefficient would encode information about the causal set discreteness scale.

### S2. The Page curve from causal set evaporation

**Idea:** Model a causal set black hole that shrinks over time (evaporation), and track the SJ entanglement entropy of the radiation region as a function of time. Does the entropy follow the Page curve (rise, peak, fall)?

**Why this is huge:** The Page curve is the central prediction of unitarity in black hole physics. The island formula (2019-2020) provided a gravitational computation of the Page curve in JT gravity. Nobody has computed it from a discrete quantum gravity model.

**Feasibility:** Low-medium. Requires: (a) a causal set that models an evaporating black hole (shrinking Schwarzschild radius), (b) the SJ vacuum on this background, (c) tracking S(radiation) as the horizon shrinks. Part (a) is the hardest — modeling evaporation dynamically on a causal set is non-trivial.

**Score if it works:** 10. This would be the first Page curve from a discrete quantum gravity model.

### S3. Derive the Einstein equations from causal set entanglement

**Idea:** Jacobson (1995) showed that if S = A/4 for all local Rindler horizons, the Einstein equations follow. We showed (B3) that the SJ vacuum on causal sets has holographic-like entanglement. Can we derive Jacobson's argument on a causal set — showing that the SJ entanglement entropy satisfies a "first law" that implies the BD action?

**Why this is huge:** This would close the circle: the BD action (a discretization of Einstein gravity) would be derived FROM the entanglement structure of the SJ vacuum, rather than being imposed as an input. Gravity would emerge from entanglement on the causal set.

**Feasibility:** Low. This is primarily a theoretical/analytical challenge, not computational. But we could test it computationally: perturb a flat causal set (add curvature), check if the entanglement entropy change δS matches the BD action variation δS_BD.

**Score if it works:** 10. This would be a deep result connecting multiple foundational ideas.

---

## Tier A: Potential 8-9 (Strong results that would attract significant attention)

### A1. Monogamy violation as a diagnostic for non-manifold causets

**Idea:** We found that the continuum phase satisfies monogamy (97%) while the crystalline phase violates it (3%). Can we use I₃ as a FILTER to select manifold-like causets from a random sample? Run a "blind test": generate causets from various sources (sprinkled, CSG, random, BD at different β), compute I₃, and check if monogamy correctly classifies manifold vs non-manifold.

**Why this is interesting:** This would transform monogamy from a diagnostic into a TOOL — a manifold-likeness criterion based purely on quantum entanglement.

**Feasibility:** High. We have all the code. Just need to run the classifier on diverse causet populations.

**Score:** 8 if the classifier works with >90% accuracy across different generation methods.

### A2. Entanglement phase diagram in the (β, ε) plane

**Idea:** Map the full phase diagram of the BD transition in the (β, ε) plane using entanglement entropy and monogamy as observables. Glaser et al. mapped the action phase diagram. Nobody has mapped the entanglement phase diagram.

**Why this is interesting:** Different ε values probe different non-locality scales. The entanglement structure might show additional phase boundaries not visible in the action.

**Feasibility:** High. Just parameter scanning — we have the code.

**Score:** 8 if new phase boundaries are found. 7 if the entanglement phase diagram matches the action phase diagram (still novel data).

### A3. Spectral dimension from the SJ propagator (not the link graph)

**Idea:** We showed (B1) that the link-graph Laplacian and the BD d'Alembertian both fail as spectral dimension probes. But the SJ Wightman function W defines a natural "heat kernel": K(σ) = exp(-σW). Compute d_s from this SJ heat kernel instead. Since W captures quantum correlations (not just connectivity), the SJ spectral dimension might correctly recover the manifold dimension.

**Why this is interesting:** This would provide the first WORKING spectral dimension estimator for causal sets — solving the open problem we identified in B1.

**Feasibility:** High. W is already computed. The heat kernel eigendecomposition is the same calculation we already do.

**Score:** 8-9 if it gives d_s = 2 for 2D sprinklings (where the link graph gives 5+). Would resolve the long-standing Eichhorn-Mizera problem constructively.

### A4. Causal set cosmology: entanglement entropy of the de Sitter horizon

**Idea:** Sprinkle a causal set into 2D de Sitter spacetime. The de Sitter horizon has a finite area (in 4D; a finite "length" in 2D). Compute the SJ entanglement entropy of the static patch (interior of the horizon). Does it satisfy S = A_horizon/(4G)?

**Why this is interesting:** The Gibbons-Hawking entropy of de Sitter space is one of the most mysterious quantities in physics. Computing it from causal set entanglement would connect our B3 results to cosmology.

**Feasibility:** Medium. We already have de Sitter sprinkling code. The challenge: defining the static patch boundary on a causal set.

**Score:** 8-9. The de Sitter entropy is a hot topic and any new computation of it attracts attention.

### A5. Machine learning manifold-likeness from the interval distribution

**Idea:** Train a neural network to classify causets as manifold-like vs non-manifold using the interval size distribution {N_n} as features. The network learns what "geometry looks like" at the level of interval statistics.

**Why this is interesting:** If the network achieves high accuracy, the learned features would tell us WHAT makes a causal set manifold-like — potentially revealing new observables. If it finds features beyond interval entropy, that's a discovery.

**Feasibility:** High. We have abundant training data (sprinkled = positive, random/KR = negative). Standard PyTorch/scikit-learn.

**Score:** 7-8 depending on what the network learns. If it discovers a new manifold-likeness criterion, 8+.

---

## Tier B: Potential 7-8 (Solid contributions to specific subfields)

### B4. Interval entropy in 3D and 4D: the full dimensional story

**Idea:** We did interval entropy in 2D (paper A) and preliminary 4D (exp31). Do it properly in 3D and 4D with finite-size scaling. The 4D non-monotonic entropy behavior was the most intriguing finding — is it an intermediate phase?

**Feasibility:** Medium-high. 3-orders and 4-orders work, just computationally expensive.

**Score:** 7 as an extension paper. 8 if the 4D intermediate phase is confirmed.

### B5. Combine B1 and B3: "What probes geometry on causal sets?"

**Idea:** A single paper that presents both the negative result (spectral dimension doesn't work) and the positive result (SJ entanglement does work) together. This is a stronger narrative than two separate papers.

**Feasibility:** High. Merge existing material.

**Score:** 8 as a combined paper (stronger than either alone).

### B6. Everpresent Lambda: predict the equation of state for DESI DR2

**Idea:** Instead of comparing with existing DESI data, make a PREDICTION for what DESI DR2 should see. The everpresent Lambda model predicts specific statistical properties of w(z) — stochastic scatter between redshift bins, specific variance, specific correlation structure. Write these down as falsifiable predictions.

**Feasibility:** High. Just need to characterize the w(z) statistics from many realizations.

**Score:** 7-8. A falsifiable prediction paper is always stronger than a comparison paper.

### B7. Entanglement entropy distinguishes 2-orders from d-orders

**Idea:** The SJ entanglement entropy should scale differently in different dimensions: S ~ ln(N) in 2D, S ~ N^{1/3} in 4D. Use this to MEASURE the effective dimension of a causal set from its entanglement properties — an "entanglement dimension estimator."

**Feasibility:** Medium. Need larger 4-orders (N=50-100) to see the scaling difference.

**Score:** 7-8. A new dimension estimator based on quantum entanglement would be useful.

---

## Tier C: Creative / Speculative (Potential 8-10 but high risk)

### C1. Causal set quantum error correction

**Idea:** In AdS/CFT, the bulk-boundary correspondence IS a quantum error correcting code (Almheiri-Dong-Harlow 2015). A causal set in the continuum phase has boundary elements (maximal/minimal) and bulk elements (interior). Is the map from bulk to boundary a quantum error correcting code? Test: delete random boundary elements and check if the bulk SJ vacuum state can be recovered from the remaining boundary.

**Why this is wild:** If true, this would mean causal set quantum gravity literally IS a quantum error correcting code — connecting directly to the Preskill program we explored in our holographic codes work.

**Feasibility:** Low-medium. The code-theoretic framework is well-defined, but "boundary" and "bulk" on a causal set are ambiguous.

**Score if it works:** 9-10.

### C2. The black hole information paradox on a causal set

**Idea:** Construct a causal set with a causal diamond that contains a "trapped region" (elements whose future light cone doesn't reach future infinity). This is a discrete black hole. Track information flow: is information about the interior accessible from the exterior via the SJ correlations? Does it satisfy the Page curve?

**Feasibility:** Low. Requires careful construction of a discrete trapped surface.

**Score:** 9-10 if it resolves any aspect of the information paradox.

### C3. Gravity from entanglement on causal sets (computational Jacobson)

**Idea:** Numerically verify Jacobson's argument on causal sets:
1. Compute SJ entanglement entropy for many small causal diamonds at different locations in a curved causal set
2. Check if the first law δS = δE_matter/(T_Unruh) holds
3. If it does, the Einstein equations (BD action) emerge from entanglement thermodynamics

This is a computational version of S3 above.

**Feasibility:** Medium. The individual computations are doable. The challenge is defining "local Rindler horizon" and "Unruh temperature" on a causal set.

**Score:** 9 if the first law is verified numerically.

### C4. Causal set complexity and the second law

**Idea:** Define circuit complexity for causal sets (how many "moves" to go from one causet to another in the 2-order space). The BD MCMC provides a natural notion of complexity = number of MCMC steps. Does complexity grow linearly with β (as the "complexity = volume" conjecture predicts for de Sitter)? Our exp29 data may already contain this information.

**Feasibility:** High — reanalyze existing MCMC trajectories.

**Score:** 7-8 if complexity = volume works. 9 if it connects to the de Sitter holography program.

---

## Meta-Strategy Recommendations

### What to do next (if optimizing for score):

1. **S1 (BH entropy from causal set entanglement)** — highest ceiling, medium feasibility. This is the moonshot. Even a partial result (wrong coefficient but correct scaling) is a 7.

2. **A3 (SJ spectral dimension)** — high ceiling, high feasibility. Directly builds on B1 and B3. Could be done in one session.

3. **B5 (Merge B1 + B3)** — not a new result but repackaging as a single coherent paper would be stronger than either alone. The narrative: "spectral dimension fails, SJ entanglement succeeds, and it's holographic."

4. **A1 (Monogamy classifier)** — quick win, high feasibility, builds directly on the strongest result we have.

### What to do next (if optimizing for novelty):

1. **C1 (Causal set QEC)** — nobody has even asked this question. If it works at all, it's instantly notable.
2. **S2 (Page curve)** — the most important open problem in quantum gravity, approached from a new angle.
3. **A4 (de Sitter entropy)** — connects to the hottest topic in the field (de Sitter holography).

### What to do next (if optimizing for effort/impact):

1. **A3** then **B5** — one session each, produces a strong combined paper at 8-9.
2. Then **A1** — quick classifier test, adds robustness to the monogamy result.
3. Then attempt **S1** — the big swing.

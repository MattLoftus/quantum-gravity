# Ideas V3: Unexplored Directions in Quantum Gravity

What we've built: causal sets (sprinkled, CSG, BD MCMC on d-orders), CDT, SJ vacuum (Wightman function, entanglement entropy, monogamy, ER=EPR), everpresent Lambda, holographic codes. ~10,000 lines of Python, 44 experiments.

What we HAVEN'T explored — organized by theme.

---

## Theme 1: Quantum Gravity Phenomenology (connects to experiment)

### 1. Lorentz invariance violation from causal set discreteness
The causal set discreteness scale l_P should produce modified dispersion relations at high energies. Compute the SJ propagator on a causal set and extract the effective dispersion relation E(p). Compare with LHAASO constraints (E_QG > 10 E_Planck for linear LIV). If the causal set gives a specific form of LIV, that's a falsifiable prediction.

**Score potential:** 8-9 (connects discrete QG to real observations)
**Feasibility:** Medium (need momentum-space SJ propagator, which requires Fourier transform on a causal set)
**Novelty:** Medium (Sorkin and others have discussed this theoretically but not computed it on MCMC-sampled causets)

### 2. Gravitational decoherence rate from the SJ vacuum
The SJ vacuum on a causal set has quantum fluctuations that could decohere macroscopic superpositions. Compute the decoherence rate for a massive particle on a causal set background. Compare with the Diosi-Penrose prediction τ_DP = ℏ/E_G. If the causal set gives a different rate, that distinguishes it from other decoherence models.

**Score potential:** 8 (connects to tabletop experiments — BMV, QGEM)
**Feasibility:** Low-Medium (need to couple the SJ vacuum to a matter sector)
**Novelty:** High

### 3. Primordial power spectrum from causal set cosmology
The everpresent Lambda gives the background cosmology. Can causal set discreteness also produce primordial perturbations? If the SJ vacuum on a de Sitter causal set produces a scale-invariant spectrum P(k) ~ k^{n_s-1} with n_s close to 0.965 (Planck measurement), that would be a major prediction.

**Score potential:** 9 (connects to CMB observations)
**Feasibility:** Low (need de Sitter SJ vacuum + perturbation theory + Fourier analysis)
**Novelty:** High (Bengochea, Leon, Perez 2025 did something similar but from discreteness, not from the SJ vacuum)

---

## Theme 2: Black Hole Physics

### 4. Unruh effect on a causal set
An accelerating observer should see a thermal bath (Unruh effect). Compute the SJ vacuum expectation value of the particle number operator along an accelerated worldline on a causal set. Does the causal set reproduce the Unruh temperature T = a/(2π)?

**Score potential:** 8 (fundamental QFT prediction tested on discrete spacetime)
**Feasibility:** Medium (need to define accelerated trajectories and particle detectors on causets)
**Novelty:** Medium (Johnston studied this theoretically; computational verification would be new)

### 5. Quantum extremal surfaces on causal sets
The island formula finds quantum extremal surfaces that determine entanglement wedges. On a causal set with a "black hole" (trapped region), can we find the island by minimizing S_gen = A/(4G) + S_bulk? We have S_bulk from the SJ vacuum. The "area" A can be defined from the causal structure.

**Score potential:** 9 (the island formula is the hottest topic in QG right now)
**Feasibility:** Low (need a proper causal set black hole + area definition)
**Novelty:** High (nobody has computed islands on causal sets)

---

## Theme 3: Random Matrix Theory Connections

### 6. Eigenvalue statistics of the Pauli-Jordan operator
The Pauli-Jordan matrix iΔ is a random antisymmetric matrix determined by the causal set sprinkling. Its eigenvalue distribution should fall into a random matrix universality class. Which one? GUE? GOE? Poisson? If the continuum phase has one universality class and the crystalline another, the phase transition is also a random matrix transition.

**Score potential:** 7-8 (connects causal sets to random matrix theory — a well-developed mathematical field)
**Feasibility:** High (we already eigendecompose iΔ — just analyze the eigenvalue statistics)
**Effort:** 1-2 hours
**Novelty:** High (nobody has studied random matrix statistics of the SJ operator)

### 7. SYK model comparison
The Sachdev-Ye-Kitaev model is a quantum mechanical model of N Majorana fermions with random all-to-all couplings. It's holographic (dual to JT gravity). The SJ vacuum on a causal set has a similar structure: random couplings determined by the causal order. Compare the eigenvalue statistics, correlation functions, and entanglement entropy of the SJ vacuum with SYK predictions.

**Score potential:** 8-9 (SYK is a major topic; any connection to causal sets would be notable)
**Feasibility:** Medium (need to implement SYK and compare)
**Novelty:** High

---

## Theme 4: Emergent Spacetime and Information

### 8. Reconstruct the metric from the SJ vacuum
The SJ Wightman function W encodes the causal structure (as we showed with ER=EPR). Can we INVERT this — reconstruct the causal set (or even the embedding coordinates) from W alone? If so, spacetime literally emerges from entanglement data.

**Score potential:** 9-10 (spacetime from entanglement — the dream of quantum gravity)
**Feasibility:** Medium (SVD or eigendecomposition of W; compare recovered structure with true causal order)
**Effort:** 2-3 hours
**Novelty:** High

### 9. Second law for SJ entanglement entropy
Define a "time direction" on a causal set (using the longest chain). Compute SJ entropy of "early" vs "late" sub-regions. Does entropy increase? If the SJ entropy satisfies a second law on causal sets, it connects to the generalized second law of black hole thermodynamics.

**Score potential:** 8 (generalized second law from discrete QG)
**Feasibility:** Medium-High (we have time-ordered elements already)
**Effort:** 2 hours
**Novelty:** High

### 10. Entanglement wedge reconstruction on causal sets
In AdS/CFT, a boundary subregion can reconstruct the bulk within its entanglement wedge. Define bulk/boundary on a causal set (e.g., interior vs maximal/minimal elements). For which boundary subregions can the bulk SJ state be recovered? Does the recoverable region match an "entanglement wedge" defined by the causal structure?

**Score potential:** 9 (entanglement wedge is central to holography)
**Feasibility:** Low-Medium (defining the wedge geometrically on a causal set is hard)
**Novelty:** High

---

## Theme 5: Multi-field and Interacting Theories

### 11. Massive scalar SJ vacuum
We've only done massless scalars. The massive SJ propagator is G_R = (1/2)C(I + m²/(2ρ)C)^{-1}. How does the mass affect entanglement entropy, monogamy, ER=EPR? Does a mass gap in the field produce a mass gap in the SJ spectrum?

**Score potential:** 6-7 (natural extension, less dramatic)
**Feasibility:** High (formula known, just a matrix inverse)
**Effort:** 1-2 hours
**Novelty:** Medium (massive SJ has been studied by Mathur & Surya 2019)

### 12. Two coupled scalar fields — entanglement between fields
Put TWO independent scalar fields on the same causal set. The vacuum state has entanglement between fields (not just between spatial regions). Compute the inter-field entanglement. Does it depend on the causal structure? This probes whether the causal set mediates entanglement between different matter species.

**Score potential:** 7-8 (novel: inter-field entanglement from causal structure)
**Feasibility:** High (just duplicate the SJ construction)
**Effort:** 2-3 hours
**Novelty:** High

---

## Theme 6: Computational / Algorithmic

### 13. Neural network SJ vacuum
Train a neural network to predict W from the causal matrix C directly, bypassing the eigendecomposition (which is O(N³)). If the network generalizes, it enables SJ computations at N=1000+ (beyond eigendecomposition limits). The architecture of the learned network might reveal what aspects of C determine W.

**Score potential:** 7 (practical speedup + interpretability)
**Feasibility:** High (standard ML)
**Effort:** 3-4 hours
**Novelty:** Medium

### 14. Quantum simulation of the BD partition function
The BD partition function Z = Σ exp(-βS) sums over causal sets. This is a combinatorial optimization problem that quantum computers could potentially speed up. Design a quantum circuit that samples from Z using quantum annealing or QAOA. Even a toy demonstration on 10-15 qubits would be a first.

**Score potential:** 7-8 (quantum computing + quantum gravity intersection)
**Feasibility:** Low-Medium (need quantum computing framework — Cirq or Qiskit)
**Novelty:** High

---

## Ranking by Expected Value

| Rank | # | Idea | Score | P(success) | EV | Effort |
|------|---|------|-------|------------|-----|--------|
| 1 | 6 | Random matrix statistics of iΔ | 7.5 | 85% | 6.4 | 1h |
| 2 | 8 | Reconstruct metric from W | 9 | 50% | 4.5 | 2h |
| 3 | 9 | Second law for SJ entropy | 8 | 60% | 4.8 | 2h |
| 4 | 11 | Massive scalar SJ vacuum | 6.5 | 85% | 5.5 | 1h |
| 5 | 4 | Unruh effect on causal set | 8 | 50% | 4.0 | 4h |
| 6 | 12 | Inter-field entanglement | 7.5 | 60% | 4.5 | 2h |
| 7 | 1 | Lorentz invariance violation | 8.5 | 40% | 3.4 | 5h |
| 8 | 7 | SYK comparison | 8.5 | 35% | 3.0 | 5h |
| 9 | 13 | Neural network SJ vacuum | 7 | 60% | 4.2 | 3h |
| 10 | 5 | Quantum extremal surfaces | 9 | 25% | 2.3 | 6h |
| 11 | 3 | Primordial power spectrum | 9 | 20% | 1.8 | 8h |
| 12 | 10 | Entanglement wedge | 9 | 20% | 1.8 | 6h |
| 13 | 2 | Gravitational decoherence | 8 | 20% | 1.6 | 6h |
| 14 | 14 | Quantum simulation of BD | 7.5 | 20% | 1.5 | 8h |

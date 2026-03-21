# Lessons Learned — Quantum Gravity Research Project

Reflections after 35 experiments, 4 papers, ~10,000 lines of code, and ~40 hours of computation.

---

## 1. How to Do Scientific Research Effectively

### Always test the null hypothesis first

The single most valuable move in this entire project was asking "does a random graph produce the same result?" before claiming any finding was physical. This saved us from publishing the d_s ≈ 2 crossing as a quantum gravity prediction (it's generic to all graphs), and from claiming the spectral dimension jump across the BD transition was meaningful (it's explained by link density alone).

**Rule: Before celebrating a result, construct the simplest possible null model and check if it reproduces the signal.** If it does, the signal is an artifact. If it doesn't, you have something. This took us from "exciting finding" to "artifact" to "real signal" multiple times, and each iteration made the work stronger.

### Calibrate against published results before doing anything new

The BD action normalization issue (factor of 4) cost us experiments 22-25 — four experiments producing data with the wrong action before we traced the discrepancy. We should have started by reproducing Surya's exact numbers (S ≈ 3.846 at β=0.1, ε=0.12, N=50) before running any new measurements.

**Rule: The first experiment with any established method should reproduce a specific published number. If it doesn't match, stop and debug before proceeding.** This is boring but prevents cascading errors.

### Negative results are often more valuable than positive ones

Our strongest contributions came from showing things DON'T work:
- CSG cannot produce spacetime (274 configurations, all fail)
- Spectral dimension on causets is determined by connectivity, not geometry
- The d_s ≈ 2 crossing is generic, not physical

Each of these saves future researchers from pursuing dead ends. A clear negative result with a proven theorem is more useful than a tentative positive result at toy scale.

### Go deep, not wide

We ran 35 experiments across 5 tracks (causal sets, CDT, holographic codes, cosmology, spectral dimension). The depth on any single track was limited — N=50-200, 2D only, simplified models. The most impactful work came when we finally committed to the BD phase transition and spent 10+ experiments going deep on one question.

**Rule: Explore broadly for one cycle to identify the most promising direction, then commit to depth.** Our first 6 experiments (broad survey) were well-spent. Experiments 7-15 (continued breadth) had diminishing returns. The project hit its stride when we focused on the BD transition (experiments 22-35).

### The research loop: compute → check artifact → if real, scale up → write

The effective research cycle turned out to be:
1. Compute the observable
2. Check against the null model (random graphs, known limits)
3. If it survives: scale up (larger N, FSS, more parameter values)
4. If it scales: connect to existing literature
5. Write the paper

Steps 2 and 4 are the ones researchers most often skip, and they're the ones that determine whether the paper is a 4/10 or a 7/10.

---

## 2. How to Think About Problems

### Ask "what is this observable actually measuring?" not "what do I want it to measure?"

We wanted spectral dimension to measure manifold dimension. It actually measures link density. We wanted the MM dimension of step-k_max=1 causets to indicate 4D manifold structure. It actually indicated a branching tree that happened to have the right pair-counting statistics.

**The observable doesn't care about your interpretation.** Figure out what it's actually sensitive to before interpreting results. The best way: test it on systems where you know the answer (sprinkled causets, lattices, random graphs).

### The most interesting questions are "why does X work?" not "does X work?"

Finding that interval entropy distinguishes the BD phases (exp26) was a 5/10 result. Understanding WHY it works — continuum gives power-law P(m) ~ m^{-(1+2/d)} while KR orders give bimodal distributions — elevated it to a 6/10. Adding the theorem that the crossing is generic (IVT proof) elevated B1 from 6 to 7. The analytic understanding is what transforms a computational observation into a scientific result.

### Cross-approach comparison reveals more than single-approach depth

The CDT vs causet comparison (exp19-20) was one of our strongest findings, because it used identical measurement code on two different quantum gravity approaches. Nobody in either community had done this — CDT people use CDT code, causet people use causet code. Using the same spectral dimension algorithm on both revealed that CDT has a growing plateau while causets don't.

**Rule: Build tools that work across approaches, then compare.** The shared codebase was the project's most underappreciated asset.

### Follow the surprise

The SJ entanglement entropy result (exp35) was the biggest surprise: CFT-like logarithmic scaling emerging from the causal set path integral without putting in conformal symmetry. This wasn't planned — it fell out of trying to implement a quantum observable for B3. The surprise signals real physics: if you expected the result, you're probably just confirming assumptions.

---

## 3. How to Push the Boundaries

### The publishable result is at the intersection of "genuinely new" and "connects to existing conversation"

Results that are new but isolated (interval entropy as an order parameter) score 5/10. Results that connect to existing published work (reinterpreting Eichhorn-Mizera through our link-density mechanism) score 7/10. The connection multiplies the impact because it gives readers a reason to care.

**Rule: After obtaining a result, search for published papers that your result reinterprets, extends, or challenges.** This takes 30 minutes and can add 2 points to the paper's score.

### Theorems > data

The IVT proof for the spectral dimension crossing took 10 minutes to write and elevated the paper more than any additional computational experiment could have. A theorem is permanent — no one needs to re-run it. Data is provisional — someone can always run at larger N or with different parameters.

**Even a trivial theorem (like the IVT application) has outsized impact in a computational paper** because it transforms an observation into a mathematical fact.

### The right normalization matters more than the right algorithm

We had the correct MCMC algorithm for the BD transition from experiment 22. But the wrong action normalization (factor of 4) made all the results unreliable. Fixing the normalization (experiments 25, 28) immediately produced clean, calibrated data. The algorithm was never the bottleneck — the calibration was.

### Knowing when to stop is a skill

Diminishing returns became visible after cycle 3 (around experiment 20). The major findings were established, and subsequent experiments were mostly confirmatory. We should have started writing papers earlier and only run additional experiments to fill specific gaps identified during writing.

---

## 4. How to Make High-Quality Papers

### The paper should tell a story, not list results

Our strongest paper (B1, spectral dimension) has a clear narrative arc:
1. Everyone says d_s → 2 is universal
2. We prove the crossing is mathematically generic (theorem)
3. We show CDT has a real plateau while causets don't (data)
4. We show why: spectral dimension tracks connectivity, not geometry (mechanism)
5. This explains the Eichhorn-Mizera anomaly (reinterpretation)

Each section raises a question that the next section answers. The weakest paper (B2, everpresent Lambda) is a collection of results without a clear arc.

### One strong claim > three medium claims

B1 makes one claim: "link-graph spectral dimension is unreliable." Everything in the paper supports this single claim from different angles. Paper A makes three claims: interval entropy is an order parameter, spectral dimension is an artifact, and there's a 4D transition. The single-claim paper is more focused and more persuasive.

### Honest caveats make papers stronger

Every limitation we disclosed (small system sizes, normalization uncertainty, truncation scheme dependence) made the paper MORE credible, not less. Referees can spot weaknesses — acknowledging them upfront signals confidence in the core result.

### Figures should be readable without the caption

A bar chart with clear colors and a reference line (fig1_peaks.pdf) communicates instantly. A dense table requires careful reading. For the key result, use a figure. For supporting data, use a table.

---

## 5. Claude Code / Memory / Workflow Improvements

### Research projects need a different memory pattern than software projects

Software projects have a clear state (what's built, what's broken, what's next). Research projects have a knowledge graph (what we tried, what we learned, what we believe). The current PLAYBOOK.md + FINDINGS.md structure worked but got unwieldy — FINDINGS.md grew to 300+ lines and became hard to navigate.

**Suggestion:** For research projects, split memory into:
- `STATE.md` — current working hypotheses, open questions, blocking issues (compact, <50 lines)
- `LOG.md` — chronological experiment log with results (append-only, can be long)
- `PLAYBOOK.md` — architecture + TODOs (as now)

The key difference from software: research STATE changes with every experiment, while software state changes with every commit. The STATE file should be rewritten (not appended) after each research cycle.

### Auto-updating reports should be triggered by milestones, not by every experiment

We updated report.html and FINDINGS.md after almost every experiment, which was noisy. Better: update after a CYCLE (3-5 experiments that answer a specific question), not after each individual run.

### The "iterate 5 times" pattern works but needs exit criteria

The user's instruction to "push → evaluate → plan, repeat 5 times" was effective, but the later iterations had diminishing returns. Better: "repeat until you hit diminishing returns OR achieve the target result, whichever comes first." The first 3 iterations produced 90% of the value.

### Subagent parallelism is powerful but fragile

Launching 3-4 research agents in parallel was our most efficient mode — one researching literature, one running experiments, one writing code. But API errors killed ~30% of agents, requiring re-launches. The pattern that worked best: launch 2-3 agents for independent tasks, do one task directly, integrate results when agents return.

### The "honest assessment" pattern is essential

Every time we stopped to honestly score a result (4/10, 5/10, 7/10) and explain WHY, it redirected effort productively. Without these checkpoints, we would have spent more time on low-impact work. The scoring rubric (what does a 5 mean? what does a 7 mean?) was essential for calibration.

**Suggestion for CLAUDE.md:** For research-oriented projects, add a rule:
> After each significant result, score it 1-10 with an honest assessment of novelty, rigor, and audience size. Use the score to decide whether to deepen or pivot.

### Context window management matters for long research sessions

This session spans multiple days and hundreds of tool calls. The context window compressed earlier conversations, losing details of experiments 1-15. For long research projects, the FINDINGS.md file serves as external memory that survives compression. Keep it structured and current — it's the primary persistence mechanism.

---

## 6. Expert Panel Peer Review Before Submission

### Run a simulated peer review before calling a paper "done"

After completing a paper draft, simulate 2-3 expert reviewers with different perspectives (e.g., domain specialist, methods expert, statistician). For each reviewer, identify specific weaknesses: missing error bars, unjustified claims, missing comparisons with prior work, unclear methodology, overclaiming.

**What we found:** Every paper had 3-5 issues that a real referee would flag. The fixes were usually small (add a caveat, clarify normalization, add a p-value) but their absence would have caused rejection or major revision. Examples:
- Paper A: missing note about action convention factor-of-4 difference with Glaser
- Paper B3: no null hypothesis for monogamy (random Gaussian states give 52%, not 0%)
- Paper B5: overclaiming "holographic" when monogamy is partition-dependent

**The process that worked:**
1. Read the full paper as each reviewer would
2. For each section, ask: "What would a skeptical expert challenge here?"
3. Fix only issues you're confident about (don't introduce new errors)
4. For issues requiring new computation (e.g., null tests), run the experiment
5. If new data weakens a claim, add caveat rather than hiding it — honesty strengthens the paper

**Key insight:** The peer review step often IMPROVED papers by forcing us to run controls we hadn't thought of (the monogamy null test, the partition-dependence test). These controls sometimes weakened claims but always produced more honest and defensible results.

---

## 7. Keep a Layman Summary

Maintain a plain-language summary (`somewhat_laymans_terms.html`) of the project and its papers. Update it when new papers are written or existing papers change significantly. This serves two purposes: (1) forces you to distill the actual insight from each paper, which often reveals whether the result is as strong as you think, and (2) provides an accessible entry point for non-specialists (including future-you who may have forgotten the details).

---

## Summary: The Meta-Rules

1. **Null hypothesis first.** Random graph control before any claim.
2. **Calibrate before exploring.** Reproduce a published number first.
3. **Go deep after one broad cycle.** Explore widely once, then commit.
4. **Connect to existing conversations.** Reinterpretation > isolated result.
5. **Theorems > data.** Even trivial proofs elevate a paper.
6. **One strong claim per paper.** Focus beats breadth.
7. **Score honestly and often.** 1-10 with reasons, used to pivot or deepen.
8. **Follow the surprise.** The unexpected result signals real physics.

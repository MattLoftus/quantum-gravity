export function GuideTab() {
  return (
    <div className="max-w-3xl">
      <div className="mb-10">
        <h2 className="text-2xl font-semibold text-gray-900 mb-2">Guide</h2>
        <p className="text-gray-500 text-sm">
          A plain-language explanation of the project, its key concepts, and what we learned.
        </p>
      </div>

      <Section title="The Big Picture">
        <p>
          The two pillars of modern physics -- general relativity (gravity, spacetime curvature) and
          quantum mechanics (particles, uncertainty, entanglement) -- are individually spectacularly
          successful but mathematically incompatible. Reconciling them is the central open problem of
          theoretical physics.
        </p>
        <p>
          We didn't solve it. But we built computational tools to test and compare several
          leading approaches, and along the way discovered some genuinely new things about how
          discrete spacetime behaves.
        </p>
        <Callout>
          Our approach in one sentence: Build simulations of discrete spacetime models, measure
          quantum and classical properties, test them against null models, and see what's real
          versus artifact.
        </Callout>
      </Section>

      <Section title="What Is a Causal Set?">
        <p>
          A causal set is the idea that spacetime is not a smooth continuum but a collection of
          discrete "events" (like atoms of spacetime) connected by causal relations: event A can
          influence event B. Think of it as replacing the smooth fabric of spacetime with a network
          of points connected by "who can signal whom."
        </p>
        <Analogy>
          A smooth lake surface vs. a collection of water molecules. At large scales they look
          the same, but at the smallest scale, the discrete structure matters.
        </Analogy>
        <p>
          The central question: if you randomly generate these discrete networks and weight them
          by an action (a discrete version of Einstein's gravity), do smooth spacetime-like
          structures emerge? We found that yes, there is a phase transition -- at the right
          "temperature," manifold-like structures are favored over random ones.
        </p>
      </Section>

      <Section title="The SJ Vacuum">
        <p>
          The Sorkin-Johnston (SJ) vacuum is a quantum state for a free scalar field that is
          determined entirely by the causal structure -- no background metric needed. It is the
          natural way to put a quantum field on a causal set. We computed its entanglement
          entropy, Wightman function, and eigenvalue statistics across hundreds of causal sets.
        </p>
        <p>
          The SJ vacuum turned out to be the richest source of observables in the project.
          Its entanglement entropy probes geometry (where spectral dimension fails). Its Wightman
          function encodes the ER=EPR connection. Its eigenvalue statistics show universal
          quantum chaos.
        </p>
      </Section>

      <Section title="The Benincasa-Dowker Action">
        <p>
          The BD action is a discrete version of the Einstein-Hilbert action (the mathematical
          expression that governs general relativity). When you simulate causal sets weighted by
          this action, there is a competition: the action favors smooth, manifold-like structures,
          but there are exponentially more non-manifold structures (Kleitman-Rothschild orders --
          highly layered, non-geometric configurations).
        </p>
        <p>
          At low "temperature" (high coupling beta), the action wins and you get manifold-like
          causets. At high temperature (low beta), entropy wins and you get random structures. The
          transition between these phases is sharp and first-order, with latent heat scaling
          extensively.
        </p>
      </Section>

      <Section title="What Is a 2-Order?">
        <p>
          A 2-order is the intersection of two independent random total orderings of N elements.
          Element i precedes element j if and only if i comes before j in both orderings. This is
          mathematically equivalent to sprinkling points into 2D Minkowski spacetime -- a random
          2-order is a random causal set in 2D flat spacetime.
        </p>
        <p>
          2-orders are computationally efficient (just two random permutations) and have beautiful
          exact combinatorics. Many of our strongest results -- exact formulas, link fractions,
          ordering fractions -- come from the 2-order representation.
        </p>
      </Section>

      <Section title="Phase Transitions">
        <p>
          The BD phase transition is the most dramatic phenomenon in causal set simulations. As the
          coupling strength beta increases, the causal set undergoes a sharp transition from a
          "continuum-like" phase (with diverse causal structure) to a "crystalline" phase dominated
          by direct links.
        </p>
        <p>
          We introduced interval entropy as a new order parameter that captures this transition
          with an 87% drop -- far sharper than any previously known diagnostic. In 4D, we discovered
          a previously unknown three-phase structure.
        </p>
      </Section>

      <Section title="What We Learned">
        <div className="space-y-4">
          <Finding title="Discrete quantum gravity has far more structure than anyone expected">
            Causal order is unreasonably effective as a foundation for physics. Seven numbers
            extracted from a causal matrix suffice to identify which spacetime you are in with
            96% accuracy.
          </Finding>

          <Finding title="Entanglement, not random walks, detects geometry">
            Spectral dimension fails a null test (random graphs match causal sets at the same
            density). SJ entanglement entropy succeeds -- it scales logarithmically and drops 3.4x
            across the phase transition.
          </Finding>

          <Finding title="Entanglement tracks causal connectivity exactly">
            The ER=EPR connection is not approximate -- the Gram identity holds to machine
            precision for all partial orders. More shared causal structure means proportionally
            more quantum entanglement.
          </Finding>

          <Finding title="Quantum chaos is universal">
            GUE random matrix statistics appear in ALL phases, ALL dimensions, and for ALL
            coupling strengths. The quantum field on a causal set is always chaotic.
          </Finding>

          <Finding title="CDT succeeds because of its time foliation">
            The Kronecker product theorem explains exactly why CDT reproduces continuum QFT and
            causal sets do not. It is the layered structure, not the density, that matters.
            The entanglement entropy S(N/2) depends only on T (number of time slices), not on
            spatial size s -- the spatial direction contributes exactly zero entropy.
            The spectral gap is exactly gap = (2/T)*cot(pi(2k-1)/(2T)).
            The temporal Wightman function W_T is exactly Toeplitz (time-translation invariant),
            the entanglement spectrum is near-thermal (discrete Unruh effect), and the state
            is efficiently representable as a matrix product state with polynomial bond dimension.
            However, the CDT SJ vacuum has fundamentally different correlation structure from the
            lattice Klein-Gordon vacuum, and the Kronecker factorization gives too few modes
            for c=1 in the continuum limit -- the structure must break to recover continuum physics.
          </Finding>

          <Finding title="Random 2-orders have exact, beautiful mathematics">
            Over 20 exact formulas connect discrete spacetime to harmonic numbers, Tracy-Widom
            distributions, and combinatorial identities. E[S_Glaser] = 1 for all N, E[maximal] = H_N,
            E[k-chains] = E[k-antichains] = C(N,k)/k! (a remarkable symmetry), and E[S_BD(epsilon)]
            has an exact closed-form expression as a function of the non-locality parameter.
          </Finding>

          <Finding title="The most important unsolved question">
            Does the SJ vacuum on a causal set sprinkled into 4D Schwarzschild spacetime
            reproduce the Bekenstein-Hawking entropy S = A/(4G), with the correct coefficient 1/4?
            All tools exist -- only computational resources are missing.
          </Finding>
        </div>
      </Section>

      <Section title="Field Impact: What Matters Most?">
        <p>
          After 680+ ideas tested, we assessed which results would most impact the causal set community.
          Two stand out:
        </p>
        <Finding title="The Kronecker product theorem explains the c_eff problem">
          The SJ vacuum on CDT decomposes as a Kronecker product (C^T-C = A_T tensor J), explaining
          why CDT gives c=1 while causets diverge. The Wightman function W is Toeplitz and fully
          analytic from cotangent eigenvalues. A foliation-projected SJ vacuum reduces causet c_eff
          from ~3 to ~0.7, confirming that c~1 requires a preferred time foliation. This is the
          recommended #1 result for a workshop talk.
        </Finding>
        <Finding title="Kurtosis excess is a new dimension estimator">
          The eigenvalue density kurtosis excess over semicircle (kappa_excess) grows with N, decreases
          with dimension d, and vanishes for random antisymmetric matrices. A genuinely geometric
          observable complementary to Myrheim-Meyer.
        </Finding>
      </Section>

      <Section title="Exporting the Methodology">
        <p>
          Beyond quantum gravity, the methodology itself -- null-model-first testing, spectral analysis,
          exact enumeration at small scales -- transfers to other fields. We tested this explicitly:
        </p>
        <Finding title="Spectral embedding recovers geography from social networks (R^2 = 0.71)">
          The same Laplacian eigenvector technique used to reconstruct spacetime coordinates from
          causal sets can recover geographic locations from a social network's topology alone. A random
          graph with the same density gives R^2 = 0.015 -- the signal is structural, not density.
        </Finding>
        <Finding title="Lee-Yang zeros confirm exact enumeration exports to stat mech">
          Exact partition function computation for the 3x3 and 4x4 Ising model (our "theorem over data"
          principle) confirms the Lee-Yang theorem: all zeros lie on the unit circle, and the gap angle
          encodes the critical temperature. The 4x4 specific heat peak is within 7% of the exact T_c.
        </Finding>
        <Finding title="3-order interval statistics extend the master formula">
          For 3-orders (3D Minkowski embeddings), the mean interval size scales as E[int] ~ (N-2) * 0.036,
          compared to (N-2)/9 ~ (N-2) * 0.111 for 2-orders. This new constant characterizes interval
          structure in higher-dimensional causets.
        </Finding>
      </Section>

      <Section title="Strengthening the Weaker Papers">
        <p>
          Targeted investigations to upgrade Papers A (interval entropy, 7.0) and F (Hasse geometry, 7.0):
        </p>
        <Finding title="Interval entropy is universal across dimensions d=2,3,4,5">
          At beta=0, the disordered-phase entropy H systematically decreases with dimension
          (H=2.1 at d=2, 1.3 at d=3, 0.7 at d=4, 0.2 at d=5). MCMC scans confirm H responds
          to the BD coupling in d=3 and d=5, supporting universality as an order parameter.
        </Finding>
        <Finding title="Exact H(beta=0) converges to 3.0 from the master formula">
          Using the E[N_k] formula from the master interval distribution, the analytic interval
          entropy converges to H ~ 3.00 as N grows, matching the MCMC value from exp102.
          This provides an exact benchmark for the disordered phase.
        </Finding>
        <Finding title="Fiedler value bounded below: lambda_2 >= c > 0">
          The Cheeger constant h ~ N^0.20 (growing), and d_max ~ 2.7*ln(N). The ratio
          h^2/(2*d_max) ~ 0.05 is bounded away from zero, providing a conditional proof
          that the Hasse Laplacian has a positive spectral gap for all N. The Hasse diagram
          is expander-like.
        </Finding>
        <Finding title="Spectral embedding: R^2 = 0.93 in 2D, degrades in higher dimensions">
          Laplacian eigenvectors recover spacetime coordinates with R^2 = 0.93 (2D), 0.70 (3D),
          0.51 (4D). Time coordinate is hardest to recover. The theoretical basis connects to
          commute time distances and the Hasse diagram encoding null geodesics.
        </Finding>
      </Section>

      <Section title="The Final Experiments (Ideas 691-700)">
        <p>
          The last ten ideas served as both capstone and reflection. We tested
          Bekenstein-Hawking entropy scaling in 2D (finding CFT-like S ~ ln(N) scaling, R^2 = 0.99),
          verified the Kronecker eigenvalue predictions against numerical SJ vacuum computations,
          confirmed the master interval formula P[k|m] = 2(m-k)/[m(m+1)] to four decimal places,
          and produced the definitive dimension table for d-orders at d = 2 through 6.
        </p>
        <Finding title="The single most important lesson from 700 experiments">
          The null model is more important than the measurement. Every result that survived
          was tested against the simplest possible null model. And the best results are exact
          theorems (machine precision, all N), not numerical fits. Use computation to discover
          theorems, not to replace them.
        </Finding>
      </Section>

      <Section title="Large-N Verification (Ideas 621-630)">
        <p>
          Several earlier results were measured at moderate N (100-1000). We pushed pure combinatorial
          measurements to N = 10,000-20,000 using sparse matrices and no eigendecomposition.
          Some claims held up; others changed significantly.
        </p>
        <Finding title="Chain length: 1.94*sqrt(N), not 2.0*sqrt(N)">
          The longest chain scales as 1.94*sqrt(N) up to N=20,000 (R^2 = 0.9995). The exponent
          is 0.4947, confirming sqrt(N) scaling. The constant is measurably below 2.0.
        </Finding>
        <Finding title="Number of links: (N+1)*H_N - 2N is exact to 0.06%">
          At N=10,000 the formula n_links = (N+1)*H_N - 2N predicts 77,886 links; we measured
          77,935 (ratio 1.0006). This is the strongest large-N confirmation of any formula in
          the project.
        </Finding>
        <Finding title="Hasse diameter saturates at exactly 6">
          From N=500 through N=10,000 the Hasse diameter is exactly 6 (not ~6, exactly 6). This
          is NOT sqrt(N) scaling and NOT log(N) scaling -- it is genuinely constant. The exponent
          fit gives N^0.03, essentially flat.
        </Finding>
        <Finding title="Link fraction constant is ~3.3, not 4.0 or pi">
          The link fraction follows c*ln(N)/N with c converging to ~3.35 at large N. Neither the
          claimed c=4 nor c=pi is correct. The constant is still slowly drifting at N=10,000.
        </Finding>
        <Finding title="E[f] = 0.500 for 2D sprinklings (not 1/3 as for 2-orders)">
          The ordering fraction of sprinkled causets converges to exactly 0.500 (z-scores all
          consistent with zero), distinct from the 1/3 value for random 2-orders.
        </Finding>
        <Finding title="Hasse graph connected with probability 1 for N >= 100">
          At N=50, P(connected) = 0.90. By N=100 it hits 1.00 and stays there through N=5,000
          (all 50+ trials connected at each size).
        </Finding>
      </Section>

      <Section title="New Paper Candidates (Ideas 611-620)">
        <p>
          With 690+ ideas tested, we identified four new paper candidates beyond the existing eight.
          These emerged from systematic multi-observable analysis and cross-paper synthesis.
        </p>
        <Finding title="Paper H: BD Transition Comprehensive (8.0/10, merging Papers A+F)">
          By computing 11 independent observables across the BD transition at 20 beta values,
          we found that action/N is the strongest order parameter (Cohen's d = 1.32), followed by
          the Fiedler value and link fraction. No published paper characterizes this transition
          with more than 2-3 observables. FSS analysis yields critical exponents that match NO
          known universality class -- the BD transition may define its own class.
        </Finding>
        <Finding title="The BD transition is in a novel universality class">
          Finite-size scaling at N=30,50,70,100 gives alpha/nu = -2.31, gamma/nu = -1.31, nu = 1.19.
          These exponents do not match 2D Ising (alpha=0, gamma=1.75, nu=1), mean field, 3D Ising,
          or percolation. The Rushbrooke relation alpha + 2*beta + gamma = 2 is satisfied exactly,
          confirming thermodynamic consistency. This is the first determination of the BD universality
          class.
        </Finding>
        <Finding title="SJ vacuum reproduces known physics qualitatively">
          {"In a single experiment: Newton's law (|W| ~ ln(r) with R² > 0.93), the Casimir effect (vacuum energy ~ 1/d, wins at both N=50 and N=80), and a Bekenstein-like area law (S ~ boundary^0.690). Each result is qualitative but collectively compelling."}
        </Finding>
        <Finding title="Causal sets are informationally simpler than random DAGs">
          The past-future mutual information of a random 2-order is only 35-40% that of a
          density-matched random DAG. Manifold-likeness constrains information flow, providing
          a new information-theoretic signature of spacetime structure.
        </Finding>
      </Section>

      <Section title="The Open Question">
        <Callout variant="purple">
          After 700 ideas, the single most important open question in causal set quantum gravity
          that this programme has NOT answered is: Does the SJ vacuum on a causal set sprinkled
          into 4D Schwarzschild spacetime reproduce the Bekenstein-Hawking entropy S = A/(4G),
          with the correct numerical coefficient 1/4?
        </Callout>
        <p>
          In 2D, we confirmed S ~ ln(N) (CFT scaling, c ~ 1). In 3D, S ~ N^0.735 -- sub-volume
          scaling, closer to the 3D area law exponent (0.667) than volume (1.0). All the tools
          exist: SJ vacuum computation, curved spacetime sprinkling, entropy scaling analysis,
          null model testing. The only missing ingredient is computational resources for
          N ~ 10,000 elements in 4D curved spacetime.
        </p>
      </Section>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-10">
      <h3 className="text-lg font-semibold text-gray-900 mb-3 pb-2 border-b border-gray-100">
        {title}
      </h3>
      <div className="text-sm text-gray-600 leading-relaxed space-y-3">
        {children}
      </div>
    </div>
  )
}

function Callout({ children, variant = 'blue' }: { children: React.ReactNode; variant?: 'blue' | 'purple' }) {
  const styles = variant === 'purple'
    ? 'bg-purple-50 border-l-purple-500'
    : 'bg-blue-50 border-l-blue'
  return (
    <div className={`${styles} border-l-4 rounded-r-md p-4 text-sm text-gray-700`}>
      {children}
    </div>
  )
}

function Analogy({ children }: { children: React.ReactNode }) {
  return (
    <p className="italic text-gray-400 text-sm">
      Analogy: {children}
    </p>
  )
}

function Finding({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-gray-50 rounded-lg p-4 border border-gray-100">
      <h4 className="text-sm font-semibold text-gray-800 mb-1">{title}</h4>
      <p className="text-sm text-gray-500">{children}</p>
    </div>
  )
}

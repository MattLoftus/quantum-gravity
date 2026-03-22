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
          </Finding>

          <Finding title="Random 2-orders have exact, beautiful mathematics">
            Exact formulas connect discrete spacetime to harmonic numbers, Tracy-Widom
            distributions, and combinatorial identities. E[S_Glaser] = 1 for all N.
          </Finding>

          <Finding title="The most important unsolved question">
            Does the SJ vacuum on a causal set sprinkled into 4D Schwarzschild spacetime
            reproduce the Bekenstein-Hawking entropy S = A/(4G), with the correct coefficient 1/4?
            All tools exist -- only computational resources are missing.
          </Finding>
        </div>
      </Section>

      <Section title="The Open Question">
        <Callout variant="purple">
          After 600 ideas, the single most important open question in causal set quantum gravity
          that this programme has NOT answered is: Does the SJ vacuum on a causal set sprinkled
          into 4D Schwarzschild spacetime reproduce the Bekenstein-Hawking entropy S = A/(4G),
          with the correct numerical coefficient 1/4?
        </Callout>
        <p>
          In 3D, we found S ~ N^0.735 -- sub-volume scaling, closer to the 3D area law exponent
          (0.667) than volume (1.0). S/A = 0.052, within an order of magnitude of the Bekenstein
          value of 0.25. All the tools exist: SJ vacuum computation, curved spacetime sprinkling,
          entropy scaling analysis, null model testing. The only missing ingredient is computational
          resources for N ~ 10,000 elements in 4D curved spacetime.
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

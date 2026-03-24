export function Hero() {
  return (
    <section className="bg-navy text-white py-20 px-4">
      <div className="max-w-3xl mx-auto text-center">
        <h1 className="text-4xl md:text-5xl font-semibold tracking-tight mb-4">
          Computational Quantum Gravity
        </h1>
        <p className="text-blue-light text-lg md:text-xl mb-8 font-light tracking-wide">
          600 experiments exploring discrete spacetime &middot; 10 papers &middot; 72,000 lines of Python
        </p>
        <p className="text-gray-300 text-base md:text-lg leading-relaxed max-w-2xl mx-auto">
          A computational research programme testing whether spacetime is fundamentally
          discrete. Using causal set theory, we built simulations of discrete spacetime models,
          measured quantum and classical properties, tested them against null models, and discovered
          that discrete quantum gravity has far more mathematical structure than anyone expected.
        </p>
        <div className="mt-10 flex flex-wrap justify-center gap-6 text-sm">
          <Stat value="600" label="Ideas tested" />
          <Stat value="~110" label="Experiments" />
          <Stat value="10" label="Papers written" />
          <Stat value="15+" label="Exact theorems" />
        </div>
      </div>
    </section>
  )
}

function Stat({ value, label }: { value: string; label: string }) {
  return (
    <div className="text-center min-w-[80px]">
      <div className="text-2xl font-semibold text-white">{value}</div>
      <div className="text-gray-400 text-xs mt-0.5">{label}</div>
    </div>
  )
}

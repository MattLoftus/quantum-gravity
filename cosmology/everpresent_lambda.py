"""
Everpresent Lambda: Stochastic Cosmological Constant from Causal Set Theory.

Implements the Ahmed, Dodelson, Greene & Sorkin (2004) model where the
cosmological constant fluctuates as Lambda ~ 1/sqrt(V_universe).

The key prediction: Lambda is not a constant but a stochastic variable
that tracks the critical density, with amplitude set by the number of
causal set elements.

Reference: Ahmed et al., Phys. Rev. D 69, 103523 (2004), arXiv:astro-ph/0209274

Normalization (the crucial fix):
- We evolve the Friedmann equation in units where H0=1, so the dimensionless
  4-volume integral V_dimless is O(1) at a=1.
- The physical 4-volume in Planck units is V_phys = V_dimless * V_H0, where
  V_H0 = (c/H0)^4 / l_P^4 ~ 10^244 is the Hubble 4-volume in Planck units.
- The number of causet elements: N = rho * V_phys, where rho ~ 1/l_P^4.
  With the comoving Hubble volume normalization, N_0 ~ 10^240 at a=1.
- The stochastic action S has dimensions of [length]^{-2} * [volume] = dimensionless.
  Lambda = S/V has dimensions of [length]^{-2} (in Planck units).
- In H0=1 units: Lambda_dimless = Lambda_phys / H0^2, so:
  Lambda_dimless = alpha * sum(xi_i * sqrt(dN_i)) / (N_total)
  ~ alpha / sqrt(N_0) ~ alpha * 10^{-120} in Planck units
  ~ alpha * (H0^2 / M_P^2) * 10^{-120+122} ~ alpha * O(1) in H0 units
  This is the Sorkin prediction: Lambda ~ H0^2 when alpha ~ O(1).

Algorithm:
1. Start with initial conditions (scale factor, matter/radiation densities)
2. At each step, compute spacetime volume V (dimensionless, H0=1 units)
3. N = V * N_0_scale (number of causet elements, properly normalized)
4. Update action: S += alpha * xi * sqrt(dN), where xi is Gaussian noise
5. Lambda_dimless = S / N (action per element, in Planck units)
   rho_Lambda = Lambda_dimless * (N_0_scale / V) to convert back to H0 units
6. Evolve Friedmann equation with this stochastic Lambda
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class CosmologyState:
    """State of the FRW universe at a given conformal time step."""
    a: float           # Scale factor
    t: float           # Cosmic time (H0^{-1} units)
    H: float           # Hubble parameter (H0 units)
    rho_m: float       # Matter density (H0^2 units)
    rho_r: float       # Radiation density (H0^2 units)
    rho_lambda: float  # Dark energy density (H0^2 units)
    Lambda: float      # Current cosmological "constant" (H0^2 units)
    S: float           # Accumulated action (Planck units)
    V: float           # Total spacetime 4-volume (dimensionless)
    N: float           # Number of causal set elements


def friedmann_H2(rho_m, rho_r, rho_lambda):
    """
    Friedmann equation: H^2 = (8*pi*G/3) * (rho_m + rho_r + rho_lambda)
    In our units where 8*pi*G/3 = 1: H^2 = rho_total
    """
    return rho_m + rho_r + rho_lambda


def compute_N0_scale():
    """
    Compute the mapping from dimensionless 4-volume (H0 units) to causet
    element count (Planck units).

    The Hubble 4-volume in Planck units:
      V_H0 = (c/H0)^4 / l_P^4

    H0 = 67.4 km/s/Mpc = 2.18e-18 s^{-1}
    c/H0 = 1.37e26 m
    l_P = 1.616e-35 m
    (c/H0)/l_P = 8.5e60

    V_H0 = (8.5e60)^4 = 5.2e243

    The comoving spatial volume within the Hubble radius:
    V_3 = (4*pi/3) * (c/H0)^3 ~ 1.08e79 m^3
    In Planck volumes: V_3 / l_P^3 ~ 2.56e182

    The 4-volume integral from a_init to a=1:
    V_4 = integral a^3(t) dt * V_3_comoving
    In dimensionless units this is O(0.5-1).

    We use N_0 ~ 10^240 as the number of elements in the past
    light cone today, following Sorkin's estimate.

    The dimensionless volume integral V_dimless(a=1) ~ 0.47 for standard
    cosmology starting from a=1e-4.

    So: N_0_scale = N_0 / V_dimless(a=1) ~ 10^240 / 0.47 ~ 2.1e240
    """
    return 2.1e240


def run_everpresent_lambda(
    alpha: float = 0.01,
    a_initial: float = 1e-4,
    a_final: float = 2.0,
    n_steps: int = 10000,
    Omega_m0: float = 0.3,
    Omega_r0: float = 9e-5,
    H0: float = 1.0,
    N0_scale: float = None,
    rng: np.random.Generator = None,
) -> list:
    """
    Evolve a FRW universe with stochastic Lambda following Ahmed et al. (2004).

    The key innovation over the previous version: proper Planck-scale normalization.
    We track N (number of causet elements) in physical Planck units while evolving
    the Friedmann equation in H0=1 units.

    Parameters:
    - alpha: coupling strength (0.01-0.02 from Ahmed et al.)
    - a_initial: initial scale factor (relative to today a=1)
    - a_final: final scale factor
    - n_steps: number of time steps
    - Omega_m0, Omega_r0: density parameters today (Omega_Lambda is NOT set;
      it emerges from the stochastic process)
    - N0_scale: elements per unit dimensionless volume (default: 2.1e240)

    Returns list of CosmologyState at each step.
    """
    if rng is None:
        rng = np.random.default_rng()
    if N0_scale is None:
        N0_scale = compute_N0_scale()

    # Today's matter/radiation densities (in units where H0 = 1)
    rho_m0 = Omega_m0 * H0 ** 2
    rho_r0 = Omega_r0 * H0 ** 2

    # Scale factors to evolve through (log-spaced for early universe resolution)
    a_values = np.logspace(np.log10(a_initial), np.log10(a_final), n_steps)

    # Initial state
    a = a_values[0]
    rho_m = rho_m0 / a ** 3
    rho_r = rho_r0 / a ** 4

    # Start with zero stochastic Lambda
    rho_lambda = 0.0
    S = 0.0          # Accumulated action (in Planck units, dimensionless number)
    V_dimless = 0.0   # Dimensionless 4-volume integral

    # Initial Hubble rate (matter+radiation dominated at early times)
    H = np.sqrt(max(0, friedmann_H2(rho_m, rho_r, rho_lambda)))
    t = 0.0

    history = []
    history.append(CosmologyState(
        a=a, t=t, H=H, rho_m=rho_m, rho_r=rho_r,
        rho_lambda=rho_lambda, Lambda=0.0, S=S, V=V_dimless, N=0.0
    ))

    for i in range(1, n_steps):
        a_new = a_values[i]
        da = a_new - a

        # Cosmic time step: dt = da / (a * H) in H0^{-1} units
        if H > 0:
            dt = da / (a * H)
        else:
            dt = da

        t += dt

        # Update matter/radiation densities
        rho_m = rho_m0 / a_new ** 3
        rho_r = rho_r0 / a_new ** 4

        # 4-volume element (dimensionless)
        # dV = a^3 * dt represents the proper 4-volume per comoving Hubble volume
        dV = a_new ** 3 * dt
        V_dimless += dV

        # Convert to causet element count
        # N = V_dimless * N0_scale gives the number of Planck-scale elements
        N = V_dimless * N0_scale
        dN = dV * N0_scale

        # Stochastic action update (Ahmed et al. 2004)
        # S_{n+1} = S_n + alpha * xi * sqrt(dN)
        # S has dimensions of [action] ~ sqrt(N) in Planck units
        if dN > 0:
            xi = rng.standard_normal()
            S += alpha * xi * np.sqrt(dN)

        # Lambda in Planck units: Lambda_P = S / N
        # Convert to H0 units: rho_lambda = Lambda_P * (H0/M_P)^2 ... but
        # more carefully:
        #
        # In Planck units: Lambda_P ~ S/N ~ alpha/sqrt(N) ~ alpha * 10^{-120}
        # rho_Lambda in Planck units = Lambda_P / (8*pi*G) = Lambda_P (since G=1 in Planck)
        #
        # To convert rho_Lambda from Planck to H0 units:
        # rho_crit_Planck = 3*H0^2/(8*pi*G) = 3*H0_Planck^2
        # H0_Planck = H0 / t_P^{-1} ~ 2.18e-18 / 1.855e43 ~ 1.18e-61
        # rho_crit_Planck ~ 3*(1.18e-61)^2 ~ 4.2e-122
        #
        # Omega_Lambda = rho_Lambda_Planck / rho_crit_Planck
        #             = Lambda_P / (3*H0_Planck^2)
        #
        # In our H0=1 units: rho_lambda = Omega_Lambda * H0^2 = Omega_Lambda
        # So: rho_lambda = Lambda_P / (3 * H0_Planck^2) * H0^2
        #
        # But we can simplify. Note:
        # Lambda_P = S / N, and S ~ alpha * sqrt(N) * (random walk)
        # So Lambda_P ~ alpha * cumulative_noise / sqrt(N) in expectation
        #
        # The cleanest way: track everything through Omega_Lambda directly.
        # Omega_Lambda = rho_Lambda / rho_crit = Lambda / (3 * H^2)
        #
        # In Planck units: Lambda_Planck = S / N
        # rho_Lambda_Planck = Lambda_Planck / (8*pi*G_Planck) = Lambda_Planck (G=1)
        #   ... actually rho = Lambda/(8piG), and in Planck units 8piG = 8pi
        #   So rho_Lambda_Planck = S / (N * 8*pi)
        #
        # Omega_Lambda = rho_Lambda / rho_crit
        #   rho_crit = 3*H^2/(8*pi*G) = 3*H^2/(8*pi) in Planck units
        #   Omega_Lambda = S / (N * 8*pi) / (3*H_Planck^2 / (8*pi))
        #               = S / (3 * N * H_Planck^2)
        #
        # H_Planck at scale factor a: H_Planck(a) = H(a) * H0_Planck (since H0=1 in our code)
        # So: Omega_Lambda = S / (3 * N * H(a)^2 * H0_Planck^2)
        #
        # But N = V_dimless * N0_scale, and we defined N0_scale = N_0 / V_dimless(a=1)
        # with N_0 ~ 10^240.
        #
        # For the Sorkin prediction to work:
        # Omega_Lambda ~ alpha * sqrt(N) / (3 * N * H^2 * H0_P^2)
        #             = alpha / (3 * sqrt(N) * H^2 * H0_P^2)
        # At a=1: N~10^240, H=1, H0_P ~ 1.18e-61
        # = alpha / (3 * 10^120 * 1 * 1.39e-122)
        # = alpha / (4.2e-2)
        # ~ 24 * alpha
        #
        # So alpha ~ 0.03 gives Omega_Lambda ~ 0.7. This is the Sorkin miracle!
        #
        # Implementation: convert S/N (Planck) to rho_lambda (H0 units)

        if N > 0:
            Lambda_planck = S / N  # Planck units

            # H0 in Planck units
            H0_planck = 1.18e-61

            # rho_lambda in H0=1 units:
            # rho_lambda = Lambda_planck / (8*pi*G) / rho_crit_H0_units
            # rho_crit in Planck = 3*H_planck^2/(8*pi)
            # rho_crit in H0 units = H^2 (by convention)
            # Omega_Lambda = Lambda_planck / (3 * H_planck^2)
            # where H_planck = H * H0_planck
            # rho_lambda (H0 units) = Omega_Lambda * H^2

            # But we need to be careful: H here is the H *including* Lambda.
            # For the update, use H from matter+radiation to avoid circular dependency:
            H_mr = np.sqrt(max(0, rho_m + rho_r))

            # Direct conversion factor from Planck Lambda to H0-unit rho:
            # rho_lambda = Lambda_planck / (3 * H0_planck^2)
            conv = 1.0 / (3.0 * H0_planck ** 2)
            rho_lambda = Lambda_planck * conv
        else:
            Lambda_planck = 0.0
            rho_lambda = 0.0

        # Update Hubble parameter with all components
        rho_total = rho_m + rho_r + rho_lambda
        if rho_total > 0:
            H = np.sqrt(rho_total)
        else:
            # Negative total energy density: universe recollapses
            H = 0.0

        a = a_new

        history.append(CosmologyState(
            a=a, t=t, H=H, rho_m=rho_m, rho_r=rho_r,
            rho_lambda=rho_lambda, Lambda=Lambda_planck,
            S=S, V=V_dimless, N=N
        ))

    return history


def run_lcdm(
    a_initial: float = 1e-4,
    a_final: float = 2.0,
    n_steps: int = 10000,
    Omega_m0: float = 0.3,
    Omega_r0: float = 9e-5,
    Omega_Lambda0: float = 0.7,
    H0: float = 1.0,
) -> list:
    """Standard LCDM evolution for comparison."""
    rho_m0 = Omega_m0 * H0 ** 2
    rho_r0 = Omega_r0 * H0 ** 2
    rho_Lambda = Omega_Lambda0 * H0 ** 2  # constant

    a_values = np.logspace(np.log10(a_initial), np.log10(a_final), n_steps)
    history = []
    t = 0.0

    for i, a in enumerate(a_values):
        rho_m = rho_m0 / a ** 3
        rho_r = rho_r0 / a ** 4
        H = np.sqrt(max(0, rho_m + rho_r + rho_Lambda))
        if i > 0:
            da = a - a_values[i - 1]
            dt = da / (a * H) if H > 0 else da
            t += dt

        history.append(CosmologyState(a=a, t=t, H=H, rho_m=rho_m, rho_r=rho_r,
                                       rho_lambda=rho_Lambda, Lambda=rho_Lambda,
                                       S=0, V=0, N=0))
    return history

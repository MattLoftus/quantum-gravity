"""
Sprinkling: generate causal sets by Poisson process into known Lorentzian manifolds.

This provides ground truth for testing dimension estimators and other observables,
since we know what manifold the causet was sprinkled into.
"""

import numpy as np
from typing import Optional
from .core import CausalSet


def sprinkle_minkowski_2d(n: int, extent_t: float = 1.0, extent_x: float = 1.0,
                          rng: Optional[np.random.Generator] = None) -> tuple:
    """
    Sprinkle n points into 2D Minkowski spacetime [0, extent_t] x [0, extent_x].

    Uses metric ds^2 = -dt^2 + dx^2.
    Two events (t1,x1) and (t2,x2) are causally related if |x2-x1| <= |t2-t1|
    (i.e., the interval is timelike or null).

    Returns (CausalSet, coordinates) where coordinates is an (n, 2) array of (t, x).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sprinkle uniformly in the causal diamond
    coords = np.zeros((n, 2))
    coords[:, 0] = rng.uniform(0, extent_t, n)  # t
    coords[:, 1] = rng.uniform(0, extent_x, n)  # x

    # Sort by time coordinate for natural labeling
    time_order = np.argsort(coords[:, 0])
    coords = coords[time_order]

    # Build causal order matrix
    causet = CausalSet(n)
    for i in range(n):
        for j in range(i + 1, n):
            dt = coords[j, 0] - coords[i, 0]
            dx = abs(coords[j, 1] - coords[i, 1])
            if dt >= dx:  # timelike or null separated
                causet.order[i, j] = 1

    return causet, coords


def sprinkle_minkowski(n: int, dim: int = 2, extent_t: float = 1.0,
                       extent_x: float = 1.0,
                       rng: Optional[np.random.Generator] = None,
                       region: str = 'diamond') -> tuple:
    """
    Sprinkle n points into d-dimensional Minkowski spacetime.

    Metric: ds^2 = -dt^2 + dx1^2 + ... + dx_{d-1}^2
    Causal relation: (dt)^2 >= sum(dxi^2)

    region='diamond': Alexandrov interval (causal diamond) between two timelike-separated
        points. This is required for the Myrheim-Meyer dimension estimator to work.
        The diamond is centered at the origin with half-height extent_t.
    region='box': rectangular region [0, extent_t] x [0, extent_x]^(d-1).

    Returns (CausalSet, coordinates) where coordinates is (n, dim).
    """
    if rng is None:
        rng = np.random.default_rng()

    if region == 'diamond':
        # Sprinkle into an Alexandrov interval (causal diamond)
        # Diamond defined by: |t| + |x| <= extent_t (in 2D)
        # In general d dimensions: |t| + r <= extent_t where r = sqrt(sum xi^2)
        # Use rejection sampling
        coords_list = []
        while len(coords_list) < n:
            # Generate candidate in bounding box
            batch_size = n * 4  # oversample for rejection
            candidates = np.zeros((batch_size, dim))
            candidates[:, 0] = rng.uniform(-extent_t, extent_t, batch_size)
            for d in range(1, dim):
                candidates[:, d] = rng.uniform(-extent_t, extent_t, batch_size)

            # Accept if inside the diamond: |t| + r <= extent_t
            r = np.sqrt(np.sum(candidates[:, 1:] ** 2, axis=1))
            mask = np.abs(candidates[:, 0]) + r <= extent_t
            accepted = candidates[mask]

            for row in accepted:
                coords_list.append(row)
                if len(coords_list) >= n:
                    break

        coords = np.array(coords_list[:n])
    else:
        coords = np.zeros((n, dim))
        coords[:, 0] = rng.uniform(0, extent_t, n)
        for d in range(1, dim):
            coords[:, d] = rng.uniform(0, extent_x, n)

    # Sort by time
    time_order = np.argsort(coords[:, 0])
    coords = coords[time_order]

    # Build causal order
    causet = CausalSet(n)
    for i in range(n):
        for j in range(i + 1, n):
            dt = coords[j, 0] - coords[i, 0]
            dx_sq = np.sum((coords[j, 1:] - coords[i, 1:]) ** 2)
            if dt * dt >= dx_sq:  # timelike or null
                causet.order[i, j] = 1

    return causet, coords


def sprinkle_de_sitter_2d(n: int, hubble: float = 1.0, eta_range: tuple = (-2.0, -0.1),
                           rng: Optional[np.random.Generator] = None) -> tuple:
    """
    Sprinkle n points into 2D de Sitter spacetime in conformal coordinates.

    Metric: ds^2 = (1/H^2 eta^2)(-d_eta^2 + dx^2)
    where eta is conformal time (negative, approaching 0 at future infinity).

    Causal structure is identical to Minkowski in conformal coordinates,
    so we sprinkle uniformly in conformal coordinates and use flat-space causality.

    Returns (CausalSet, conformal_coordinates).
    """
    if rng is None:
        rng = np.random.default_rng()

    coords = np.zeros((n, 2))
    coords[:, 0] = rng.uniform(eta_range[0], eta_range[1], n)  # conformal time
    coords[:, 1] = rng.uniform(0, 2 * np.pi / hubble, n)  # spatial

    # Sort by conformal time
    time_order = np.argsort(coords[:, 0])
    coords = coords[time_order]

    # Causal structure in conformal coords is the same as Minkowski
    causet = CausalSet(n)
    for i in range(n):
        for j in range(i + 1, n):
            d_eta = coords[j, 0] - coords[i, 0]  # positive since sorted
            dx = abs(coords[j, 1] - coords[i, 1])
            if d_eta >= dx:
                causet.order[i, j] = 1

    return causet, coords

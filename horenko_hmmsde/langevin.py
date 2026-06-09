"""
Euler-Maruyama integrator for overdamped Langevin dynamics

    dX_t = - grad V(X_t) dt + sqrt(2 eps) dW_t.

Returns a thinned trajectory:  the integrator runs at the (small) `dt_sim`
step, but we only store every `thin`-th sample, so the observed
discrete-time process has lag tau = dt_sim * thin.  This is the time step
the HMM-SDE M-step then sees.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from .potential import Potential


@dataclass
class Trajectory:
    X: np.ndarray            # (T, dim) observed positions
    t: np.ndarray            # (T,)
    tau: float               # observation lag = dt_sim * thin
    dt_sim: float            # integrator step
    eps: float
    potential_name: str


def simulate_overdamped(
    potential: Potential,
    eps: float,
    T_phys: float,
    dt_sim: float = 1e-3,
    thin: int = 100,
    x0: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Trajectory:
    """
    Generate an overdamped Langevin trajectory by Euler-Maruyama.

    The returned trajectory has lag tau = dt_sim * thin.  In all the
    HMM-SDE downstream code, "tau" refers to this observation lag.

    Parameters
    ----------
    potential : Potential
    eps : float, noise intensity.
    T_phys : float, total physical time of the simulation.
    dt_sim : float, integration step (must be small enough for stability
        on the chosen potential; for scaled MB at scale=0.01, dt=1e-3 is safe).
    thin : int, store every thin-th step.
    x0 : (dim,) initial position; defaults to the deepest minimum.
    seed : int, RNG seed.
    """
    rng = np.random.default_rng(seed)
    dim = potential.minima[0].position.size

    if x0 is None:
        x0 = potential.minima[0].position.copy()
    x = np.asarray(x0, dtype=float).copy()

    n_steps = int(round(T_phys / dt_sim))
    n_store = n_steps // thin + 1

    X_out = np.empty((n_store, dim))
    t_out = np.empty(n_store)
    X_out[0] = x
    t_out[0] = 0.0

    sigma = np.sqrt(2.0 * eps * dt_sim)
    store_idx = 1
    for i in range(1, n_steps + 1):
        g = potential.grad_V(x)
        x = x - g * dt_sim + sigma * rng.standard_normal(dim)
        if i % thin == 0 and store_idx < n_store:
            X_out[store_idx] = x
            t_out[store_idx] = i * dt_sim
            store_idx += 1

    X_out = X_out[:store_idx]
    t_out = t_out[:store_idx]

    return Trajectory(
        X=X_out, t=t_out,
        tau=dt_sim * thin,
        dt_sim=dt_sim,
        eps=eps,
        potential_name=potential.name,
    )


def assign_well(X: np.ndarray, potential: Potential) -> np.ndarray:
    """
    Hard well assignment by nearest minimum (ground-truth diagnostic).
    """
    centres = np.stack([m.position for m in potential.minima])  # (M, dim)
    d2 = np.sum((X[:, None, :] - centres[None, :, :]) ** 2, axis=2)
    return np.argmin(d2, axis=1)

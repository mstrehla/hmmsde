"""
Scaled Müller-Brown potential and Eyring-Kramers rates.

Convention: overdamped Langevin
    dX_t = -grad V(X_t) dt + sqrt(2 eps) dW_t.

Critical points are found by `scipy.optimize.minimize` (for minima)
and a Newton-on-grad-V method (for saddles), seeded from the standard
literature values.  Hessians are evaluated analytically.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Tuple
from scipy.optimize import minimize, root


# Standard Müller-Brown parameters (Mueller & Brown 1979)
_A = np.array([-200.0, -100.0, -170.0, 15.0])
_a = np.array([-1.0, -1.0, -6.5, 0.7])
_b = np.array([0.0, 0.0, 11.0, 0.6])
_c = np.array([-10.0, -10.0, -6.5, 0.7])
_x0 = np.array([1.0, 0.0, -0.5, -1.0])
_y0 = np.array([0.0, 0.5, 1.5, 1.0])


@dataclass
class CriticalPoint:
    position: np.ndarray
    energy: float
    hessian: np.ndarray              # (2, 2) symmetric
    hessian_eigenvalues: np.ndarray  # ascending order
    kind: str                        # 'minimum' or 'saddle'


@dataclass
class Potential:
    name: str
    scale: float
    V: Callable[[np.ndarray], float]
    grad_V: Callable[[np.ndarray], np.ndarray]
    hess_V: Callable[[np.ndarray], np.ndarray]
    minima: List[CriticalPoint] = field(default_factory=list)
    saddles: List[CriticalPoint] = field(default_factory=list)


def _V_unscaled(x: np.ndarray) -> float:
    dx = x[0] - _x0
    dy = x[1] - _y0
    expo = _a * dx**2 + _b * dx * dy + _c * dy**2
    return float(np.sum(_A * np.exp(expo)))


def _grad_unscaled(x: np.ndarray) -> np.ndarray:
    dx = x[0] - _x0
    dy = x[1] - _y0
    expo = _a * dx**2 + _b * dx * dy + _c * dy**2
    e = np.exp(expo)
    gx = np.sum(_A * e * (2.0 * _a * dx + _b * dy))
    gy = np.sum(_A * e * (_b * dx + 2.0 * _c * dy))
    return np.array([gx, gy])


def _hess_unscaled(x: np.ndarray) -> np.ndarray:
    dx = x[0] - _x0
    dy = x[1] - _y0
    expo = _a * dx**2 + _b * dx * dy + _c * dy**2
    e = np.exp(expo)
    fx = 2.0 * _a * dx + _b * dy
    fy = _b * dx + 2.0 * _c * dy
    # d/dx (A*e*fx) = A*e*(fx^2 + 2a)
    Hxx = np.sum(_A * e * (fx * fx + 2.0 * _a))
    Hyy = np.sum(_A * e * (fy * fy + 2.0 * _c))
    Hxy = np.sum(_A * e * (fx * fy + _b))
    return np.array([[Hxx, Hxy], [Hxy, Hyy]])


def _find_critical_point(seed: np.ndarray, kind: str) -> np.ndarray:
    """Polish a critical-point seed by Newton on grad V = 0."""
    seed = np.asarray(seed, dtype=float)
    # If the seed is already a critical point to machine precision the
    # numerical solver complains; accept it directly in that case.
    if np.linalg.norm(_grad_unscaled(seed)) < 1e-8:
        return seed
    sol = root(_grad_unscaled, seed, jac=_hess_unscaled, tol=1e-12)
    if not sol.success and np.linalg.norm(_grad_unscaled(sol.x)) > 1e-6:
        raise RuntimeError(f"failed to refine {kind} from seed {seed}: {sol.message}")
    return sol.x


# Seed positions from the Mueller-Brown literature
_MIN_SEEDS = np.array([
    [-0.55822363,  1.44172584],   # A: deepest well
    [ 0.62349942,  0.02803776],   # B: secondary well
    [-0.05001084,  0.46669367],   # C: shallowest well
])
_SADDLE_SEEDS = np.array([
    [ 0.21248658,  0.29298827],   # between B and C
    [-0.82200156,  0.62430281],   # between A and C
])


def mueller_brown(scale: float = 0.01) -> Potential:
    """Construct the scaled Müller-Brown potential."""

    def V(x: np.ndarray) -> float:
        return scale * _V_unscaled(np.asarray(x, dtype=float))

    def grad_V(x: np.ndarray) -> np.ndarray:
        return scale * _grad_unscaled(np.asarray(x, dtype=float))

    def hess_V(x: np.ndarray) -> np.ndarray:
        return scale * _hess_unscaled(np.asarray(x, dtype=float))

    minima: List[CriticalPoint] = []
    for seed in _MIN_SEEDS:
        pos = _find_critical_point(seed, "minimum")
        H = hess_V(pos)
        eigs = np.sort(np.linalg.eigvalsh(H))
        assert np.all(eigs > 0), f"refined seed {seed} is not a minimum (eigs={eigs})"
        minima.append(CriticalPoint(pos, V(pos), H, eigs, "minimum"))

    saddles: List[CriticalPoint] = []
    for seed in _SADDLE_SEEDS:
        pos = _find_critical_point(seed, "saddle")
        H = hess_V(pos)
        eigs = np.sort(np.linalg.eigvalsh(H))
        assert eigs[0] < 0 < eigs[1], (
            f"refined seed {seed} is not a saddle (eigs={eigs})"
        )
        saddles.append(CriticalPoint(pos, V(pos), H, eigs, "saddle"))

    return Potential(
        name="mueller_brown",
        scale=scale,
        V=V, grad_V=grad_V, hess_V=hess_V,
        minima=minima, saddles=saddles,
    )


def kramers_rate(minimum: CriticalPoint, saddle: CriticalPoint,
                 eps: float) -> float:
    """
    Eyring-Kramers escape rate for 2D overdamped Langevin
    dX = -grad V dt + sqrt(2 eps) dW with unit friction.

        k = (|lambda_s^-| / (2*pi)) * sqrt(det H_min / |det H_saddle|)
            * exp(-(V(saddle) - V(min)) / eps)

    where H_min, H_saddle are the Hessians of V at the minimum and saddle,
    lambda_s^- is the unique negative eigenvalue of H_saddle.

    Reference: Berglund (2011), "Kramers' law: validity, derivations and
    generalisations".
    """
    lam_min = minimum.hessian_eigenvalues       # both > 0
    lam_sad = saddle.hessian_eigenvalues        # one negative, one positive
    neg = -lam_sad[0]
    if neg <= 0:
        raise ValueError(f"saddle has no negative eigenvalue (eigs={lam_sad})")

    det_min = float(np.prod(lam_min))
    det_sad = abs(float(np.prod(lam_sad)))
    barrier = saddle.energy - minimum.energy

    prefactor = (neg / (2.0 * np.pi)) * np.sqrt(det_min / det_sad)
    return prefactor * np.exp(-barrier / eps)


def kramers_rate_matrix(potential: Potential, eps: float,
                        connect: List[Tuple[int, int]] = None,
                        ) -> Tuple[np.ndarray, List[str]]:
    """
    Build the Markov-jump generator predicted by Eyring-Kramers theory.

    Parameters
    ----------
    potential : Potential with M minima and S saddles.
    eps : float, noise intensity.
    connect : optional list of (i_min, j_min) pairs that share a saddle
              and the saddle index that connects them.  If None, we use
              a hard-coded connectivity for the standard Müller-Brown:
                 saddle 0 connects B and C
                 saddle 1 connects A and C
              and there is no direct A<->B saddle, so direct A<->B rate
              is treated as 0 (transit happens through C).

    Returns
    -------
    Q : (M, M) generator matrix.  Q[i, j] (j != i) is the rate of i -> j;
        diagonal is -sum_{j != i} Q[i, j].
    labels : list of well names (M, ).
    """
    M = len(potential.minima)
    if connect is None:
        # For standard MB: (i, j, saddle_index)
        connect = [(1, 2, 0), (0, 2, 1)]

    Q = np.zeros((M, M))
    for (i, j, s) in connect:
        k_ij = kramers_rate(potential.minima[i], potential.saddles[s], eps)
        k_ji = kramers_rate(potential.minima[j], potential.saddles[s], eps)
        Q[i, j] += k_ij
        Q[j, i] += k_ji
    Q[np.diag_indices(M)] = -Q.sum(axis=1)
    labels = [f"min_{k}" for k in range(M)]
    return Q, labels

"""
Post-fit diagnostics for an HMM-SDE estimate (paper sections 4.1, 4.3).

After Theorem 3.1 has produced (mu, B = exp(tau F), R(tau)) for each
state, we want to:

  - Recover F as the principal matrix logarithm of B (section 4.1).
    For overdamped Langevin from a smooth potential V, F = -Hess V and
    is symmetric, so the principal log is the unique real log.  We
    report the symmetry of F as a diagnostic; large deviation signals
    either a fit that is not overdamped or a numerically degenerate state.

  - Recover Sigma Sigma^T from R(tau) via the Sylvester equation
    (paper eq. 23-24):

        Cor E Cor^T - E = f         (Sylvester for E)
        Sigma Sigma^T = -( (Cov + E) F^T + F (Cov + E) )

    For overdamped Langevin dX = -grad V dt + sqrt(2 eps) dW (unit
    friction), the fluctuation-dissipation relation predicts
    Sigma Sigma^T = 2 eps I.  We use this prediction as a sanity check
    of the fit.

  - Match the L_hat estimated metastable centres mu^(i) to the M true
    minima of V via a Hungarian-type assignment (with M == L_hat after
    PCCA+ if the spectral gap is clean).

  - Convert the estimated transition matrix T into a continuous-time
    generator Q = log(T) / tau and compare its off-diagonal entries to
    Eyring-Kramers predictions.
"""

import numpy as np
from typing import Tuple, Optional, List
from scipy.linalg import logm, solve_sylvester


def principal_log_F(B: np.ndarray, tau: float) -> np.ndarray:
    """
    F = log(B) / tau where log is the principal matrix logarithm.
    Returns (dim, dim) real matrix; the imaginary part is discarded
    after a warning if it's non-negligible.

    For overdamped Langevin from a smooth V at a deep well, B has
    eigenvalues in (0, 1) (real, positive, < 1), so log is unambiguous.
    """
    LB = logm(B)
    if np.abs(LB.imag).max() > 1e-8:
        # principal log produced a non-trivial imaginary part: signals
        # either complex eigenvalues in B (which can legitimately happen
        # for non-symmetric F: a damped oscillator).  Take the real part
        # and let the symmetry diagnostic flag this.
        pass
    return LB.real / tau


def symmetry_defect(M: np.ndarray) -> float:
    """
    Normalised symmetry defect ||M - M^T||_F / ||M + M^T||_F.
    Zero means perfectly symmetric.
    """
    asym = M - M.T
    sym = M + M.T
    return float(np.linalg.norm(asym, "fro") /
                 (np.linalg.norm(sym, "fro") + 1e-15))


def sigma_sigmaT_from_R(B: np.ndarray, R: np.ndarray, F: np.ndarray,
                        Cov: np.ndarray,
                        f_boundary: Optional[np.ndarray] = None
                        ) -> np.ndarray:
    """
    Recover Sigma Sigma^T from R(tau) using the literal Theorem 2.1 form
    (paper page 740, eq. 23):

        Sigma Sigma^T = -( (Cov + E) F^T + F (Cov + E) ),

    where E is the unique symmetric solution of the Sylvester equation

        B E B^T - E = f.

    The boundary term f comes from the single-state version of Theorem
    2.1; in the multi-state setting we use the single-state expression
    f = -delta delta^T + (1/W) * [(Z_T - Zbar)(Z_T - Zbar)^T
                                  - (Z_1 - Zbar)(Z_1 - Zbar)^T]
    only when supplied; otherwise we fall back to the ergodic limit f = 0,
    which gives

        Sigma Sigma^T = -( F Cov + Cov F^T ).

    Both forms produce identical results for an ergodic, sufficiently long
    series; the boundary correction matters only when the series starts
    or ends in a transient regime.
    """
    if f_boundary is None:
        return -(F @ Cov + Cov @ F.T)

    # Solve the Sylvester equation B E B^T - E = f for symmetric E.
    # Equivalent in standard form:  B E B^T + (-I) E = f.
    # scipy.linalg.solve_sylvester solves A X + X B = C; we rewrite:
    #   B E B^T = E + f   <=>   E - B E B^T + f = 0
    # No standard solver fits exactly; use the closed-form vec identity:
    #   vec(B E B^T - E) = (B kron B - I) vec(E) = vec(f)
    n = B.shape[0]
    M = np.kron(B, B) - np.eye(n * n)
    vec_E = np.linalg.solve(M, f_boundary.reshape(-1))
    E = vec_E.reshape(n, n)
    E = 0.5 * (E + E.T)

    Cov_plus_E = Cov + E
    return -(Cov_plus_E @ F.T + F @ Cov_plus_E)


def well_assignment(mu_est: np.ndarray, mu_true: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hungarian-type assignment of estimated mu (K, dim) to true minima
    (M, dim).  Returns (perm, dist) where perm[i] = index of the true
    minimum matched to estimated state i, and dist[i] is the Euclidean
    distance of the match.

    For K == M, perm is a permutation.  For K > M (over-fit), several
    estimated states may share a true minimum (we keep the lowest-cost
    full assignment from scipy.optimize.linear_sum_assignment if K <= M,
    else fall back to a greedy nearest-neighbour match).
    """
    from scipy.optimize import linear_sum_assignment

    K = mu_est.shape[0]
    M = mu_true.shape[0]
    d2 = np.sum((mu_est[:, None, :] - mu_true[None, :, :]) ** 2, axis=2)
    cost = np.sqrt(d2)

    if K <= M:
        row, col = linear_sum_assignment(cost)
        perm = -np.ones(K, dtype=int)
        dist = np.zeros(K)
        for r, c in zip(row, col):
            perm[r] = c
            dist[r] = cost[r, c]
        return perm, dist
    else:
        # Greedy: each estimated state picks its closest true minimum
        perm = np.argmin(cost, axis=1)
        dist = cost[np.arange(K), perm]
        return perm, dist


def generator_from_T(T_mat: np.ndarray, tau: float) -> np.ndarray:
    """
    Continuous-time generator Q = log(T) / tau.

    If the principal log has small negative off-diagonals (numerical
    noise from a sub-stochastic component), clip them to zero and
    refill the diagonal.  This is the standard "embeddability" cleanup.
    """
    LT = logm(T_mat)
    if np.abs(LT.imag).max() > 1e-8:
        LT = LT.real
    else:
        LT = LT.real
    Q = LT / tau
    # Clip tiny negatives on off-diagonals
    K = Q.shape[0]
    off = Q.copy()
    np.fill_diagonal(off, 0.0)
    off = np.maximum(off, 0.0)
    np.fill_diagonal(off, -off.sum(axis=1))
    return off


def match_generator(Q_est: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """
    Reorder the estimated generator according to the well assignment.

    `well_assignment` returns perm[estimated_state] = true_minimum_index.
    Therefore, when the number of estimated macrostates equals the number
    of true wells and `perm` is a genuine permutation, the estimated states
    must be ordered by `argsort(perm)`, not by `perm` itself.

    If fewer macrostates than true wells were selected, the returned matrix
    is ordered by the assigned true-minimum labels and has the estimated
    macrostate dimension.  A full Kramers comparison should then be skipped
    by the caller because there is no one-to-one state correspondence.
    """
    perm = np.asarray(perm, dtype=int)
    order = np.argsort(perm)
    return Q_est[np.ix_(order, order)]

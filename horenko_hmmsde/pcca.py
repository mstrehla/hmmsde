"""
PCCA+ (robust Perron Cluster Cluster Analysis) for lumping a fine-grained
HMM transition matrix into metastable macro-states.

We use the inner simplex algorithm of Roeblitz & Weber (2013):

  1.  Eigendecompose the row-stochastic transition matrix T.  Sort
      eigenvalues by descending real part; the leading eigenvalue is 1
      (with constant right-eigenvector).
  2.  Take the first L_hat right-eigenvectors as columns of X (L x L_hat).
      The L rows of X live inside an (L_hat - 1)-simplex; the vertices of
      that simplex correspond to the L_hat "purest" metastable states.
  3.  Greedy inner-simplex search to identify L_hat vertex rows.
  4.  Membership matrix chi = X @ A is the affine transformation that
      maps the vertex rows to the standard simplex (rows sum to 1, each
      entry in [0, 1]).

We pick L_hat using the implied-timescale spectral gap of T:
   its_i = -tau / log(|lambda_i|),
and the rule "L_hat is the smallest i with its_i / its_{i+1} > threshold".
A user override is honoured for reproducibility.
"""

import numpy as np
from typing import Tuple, Optional, List


def sort_eigendecomposition(T_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eigendecomposition of a row-stochastic matrix, sorted by descending
    real part of the eigenvalues.  Returns (eigenvalues, right_eigvecs).
    Numerical imaginary parts are taken as zero (real Perron spectrum
    is the regime we care about).
    """
    vals, vecs = np.linalg.eig(T_mat)
    real_vals = vals.real
    real_vecs = vecs.real
    order = np.argsort(-real_vals)
    return real_vals[order], real_vecs[:, order]


def implied_timescales(eigenvalues: np.ndarray, tau: float) -> np.ndarray:
    """
    its_i = -tau / log(|lambda_i|).  Returns inf for lambda_i = 1, zero
    for |lambda_i| <= 0.
    """
    out = np.empty_like(eigenvalues, dtype=float)
    for i, lam in enumerate(eigenvalues):
        absl = abs(lam)
        if absl <= 0:
            out[i] = 0.0
        elif absl >= 1.0:
            out[i] = np.inf
        else:
            out[i] = -tau / np.log(absl)
    return out


def select_macrostate_count(eigenvalues: np.ndarray, tau: float,
                            min_K: int = 2, max_K: Optional[int] = None,
                            gap_threshold: float = 2.0,
                            fast_floor_factor: float = 10.0,
                            verbose: bool = False
                            ) -> Tuple[int, np.ndarray]:
    """
    Pick L_hat from the implied-timescale gap.  Returns (L_hat, its).

    Rule: scan i = min_K - 1, ..., max_K - 1.  The gap at position i
    separates "slow" timescales {its[0], ..., its[i]} from "fast"
    timescales {its[i+1], ...}.  We accept the *first* i satisfying
        its[i] / its[i+1] > gap_threshold       (significant gap)
        its[i+1] < fast_floor_factor * tau      (faster side is genuinely fast)
    This implements the standard "the smallest k beyond which the
    process equilibrates within tau" heuristic, in contrast to taking
    the largest absolute gap (which can be triggered by a near-zero
    eigenvalue purely from numerical decay).

    If no qualifying gap is found, fall back to min_K and emit a warning.
    """
    L = len(eigenvalues)
    if max_K is None:
        max_K = L
    its = implied_timescales(eigenvalues, tau)

    fast_thresh = fast_floor_factor * tau
    for i in range(min_K - 1, min(max_K - 1, L - 1)):
        if its[i + 1] <= 0 or its[i + 1] == np.inf:
            continue
        ratio = its[i] / its[i + 1]
        meets_gap = ratio > gap_threshold
        is_fast_below = its[i + 1] < fast_thresh
        if verbose:
            print(f"  its[{i}]={its[i]:.3f}, its[{i+1}]={its[i+1]:.3f}, "
                  f"ratio={ratio:.2f}, fast_below={is_fast_below}")
        if meets_gap and is_fast_below:
            return i + 1, its

    if verbose:
        print(f"  no qualifying gap found (threshold {gap_threshold}, "
              f"fast floor {fast_thresh:.3f}); falling back to min_K={min_K}")
    return min_K, its


def _inner_simplex_vertices(X: np.ndarray) -> List[int]:
    """
    Identify L_hat vertex rows of X via greedy inner-simplex search.

    The L_hat rows of X selected by this procedure span the largest
    simplex (in terms of volume) inside the row-cloud of X.

    Strategy (Weber 2006, Roeblitz & Weber 2013):
      1.  First vertex: row of X farthest from the origin (in the
          subspace orthogonal to the constant first column, which is
          the eigenvector of eigenvalue 1).  Equivalently, the row
          farthest from the centroid.
      2.  Subsequent vertices: row farthest (in Euclidean distance)
          from the affine span of already-chosen vertices.
    """
    L, L_hat = X.shape

    # Centre the rows (remove the constant first eigenvector contribution
    # by subtracting the mean).  In practice the first eigenvector of a
    # row-stochastic matrix is a constant; sub-spaces orthogonal to it
    # capture the metastable structure.
    Xc = X - X.mean(axis=0, keepdims=True)

    # First vertex: row with the largest norm in centered coords.
    norms = np.linalg.norm(Xc, axis=1)
    v0 = int(np.argmax(norms))
    vertices = [v0]

    # Build an orthonormal basis incrementally.  basis stores the
    # (L_hat - len(vertices) + 1) orthonormal directions; the residual
    # of row i is what's left after projection onto span(vertices).
    while len(vertices) < L_hat:
        # Anchor point = first vertex; differences w.r.t. anchor
        anchor = Xc[vertices[0]]
        if len(vertices) == 1:
            # Distance from each row to anchor
            d = np.linalg.norm(Xc - anchor, axis=1)
        else:
            # Affine span of vertices: anchor + span(v_j - anchor) for j>0
            V = Xc[vertices[1:]] - anchor          # (m, L_hat)
            # Orthonormalise V's row space
            Q, _ = np.linalg.qr(V.T)               # (L_hat, m)
            # Residual of row i: (row_i - anchor) - proj
            diff = Xc - anchor                     # (L, L_hat)
            proj = diff @ Q @ Q.T                  # (L, L_hat)
            resid = diff - proj
            d = np.linalg.norm(resid, axis=1)
        # Mask already-chosen vertices
        d_masked = d.copy()
        d_masked[vertices] = -1.0
        vertices.append(int(np.argmax(d_masked)))

    return vertices


def _pcca_embedding(T_mat: np.ndarray, L_hat: int
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the PCCA+ eigenvector embedding.

    For a row-stochastic transition matrix the Perron right eigenvector is
    the constant vector, while the stationary distribution is the Perron
    left eigenvector.  PCCA+ in this row-stochastic convention therefore
    uses right eigenvectors and normalises the first column to exactly one.

    The previous implementation selected the simplex vertices before this
    normalisation.  Since the inner-simplex search should see the same
    coordinates that are later mapped to the standard simplex, we normalise
    first and only then choose vertices.
    """
    eigvals, eigvecs = sort_eigendecomposition(T_mat)
    X = eigvecs[:, :L_hat].copy()

    if not np.allclose(T_mat.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("PCCA+ expects a row-stochastic transition matrix")

    X[:, 0] = 1.0

    return eigvals, X


def pcca_plus(T_mat: np.ndarray, L_hat: int
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run PCCA+ on transition matrix T to produce a membership matrix.

    Returns
    -------
    chi : (L, L_hat) membership matrix.  chi[i, j] in [0, 1], rows sum to 1.
    eigenvalues : sorted eigenvalues of T (descending).
    vertices : list of L_hat micro-state indices selected as vertices.
    """
    L = T_mat.shape[0]
    if L_hat == L:
        return np.eye(L), np.linalg.eigvals(T_mat).real, list(range(L))
    if L_hat < 1:
        raise ValueError(f"L_hat must be >= 1, got {L_hat}")

    eigvals, X = _pcca_embedding(T_mat, L_hat)
    vertices = _inner_simplex_vertices(X)

    # Solve for A: X @ A maps vertex rows to identity (standard simplex).
    # X[vertices] @ A = I  =>  A = X[vertices]^{-1}
    V_block = X[vertices]                     # (L_hat, L_hat)
    try:
        A = np.linalg.inv(V_block)
    except np.linalg.LinAlgError:
        A = np.linalg.pinv(V_block)
    chi = X @ A                               # (L, L_hat)

    # Clip tiny negatives (numerical noise) and renormalise
    chi = np.maximum(chi, 0.0)
    rs = chi.sum(axis=1, keepdims=True) + 1e-15
    chi = chi / rs
    return chi, eigvals, vertices


def aggregate_params(params, chi: np.ndarray, nu: np.ndarray,
                     T_mat: np.ndarray,
                     Z: Optional[np.ndarray] = None,
                     tau: Optional[float] = None):
    """
    Aggregate the K' micro-state HMM-SDE parameters into L_hat macro-state
    parameters using the membership matrix chi (paper step 6).
    """
    from .hmmsde import HMMSDEParams, _m_step_horenko

    L_micro, L_hat = chi.shape
    dim = params.dim

    # Macro stationary distribution and macro transition matrix (same in
    # both routes; pure projection of the micro objects).
    pi_micro = nu.mean(axis=0)
    pi_macro = chi.T @ pi_micro
    pi_macro = pi_macro / pi_macro.sum()
    D_pi = np.diag(pi_micro)
    T_unnorm = chi.T @ D_pi @ T_mat @ chi
    T_macro = T_unnorm / (T_unnorm.sum(axis=1, keepdims=True) + 1e-15)

    if Z is not None and tau is not None:
        # Route (b): M-step on macro posterior
        nu_macro = np.clip(nu @ chi, 0.0, 1.0)
        nu_macro = nu_macro / (nu_macro.sum(axis=1, keepdims=True) + 1e-15)
        m_out = _m_step_horenko(Z, nu_macro, np.eye(L_hat), tau)
        return HMMSDEParams(K=L_hat, dim=dim, tau=tau,
                            mu=m_out.mu, B=m_out.B, R=m_out.R,
                            T=T_macro, pi=pi_macro)

    # Route (a): chi-weighted aggregation
    W = pi_micro[:, None] * chi
    W_J = W.sum(axis=0) + 1e-15
    mu = (W.T @ params.mu) / W_J[:, None]
    B = np.einsum("iJ,ijk->Jjk", W, params.B) / W_J[:, None, None]
    R = np.einsum("iJ,ijk->Jjk", W, params.R) / W_J[:, None, None]
    return HMMSDEParams(K=L_hat, dim=dim, tau=params.tau,
                        mu=mu, B=B, R=R, T=T_macro, pi=pi_macro)

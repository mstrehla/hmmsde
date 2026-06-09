"""
HMM-SDE estimation following Horenko & Schuette.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from scipy.linalg import solve_triangular
import warnings


@dataclass
class HMMSDEParams:
    K: int                  # number of hidden states
    dim: int                # dimension of observed Z
    tau: float              # observation lag
    mu: np.ndarray          # (K, dim)
    B: np.ndarray           # (K, dim, dim)  = exp(tau F)
    R: np.ndarray           # (K, dim, dim)  residual covariance per state
    T: np.ndarray           # (K, K) transition matrix of the HMM
    pi: np.ndarray          # (K,)


@dataclass
class HMMSDEResult:
    params: HMMSDEParams
    nu: np.ndarray                  # (T, K) smoothed posterior P(z_t = i | Z)
    log_likelihood: float
    log_likelihood_history: List[float]
    bic: float
    viterbi: np.ndarray             # (T,) most-likely state sequence
    n_iter: int
    converged: bool


# ------------------------------------------------------------------
# E-step: scaled forward-backward
# ------------------------------------------------------------------
def _emission_loglik(Z: np.ndarray, params: HMMSDEParams) -> np.ndarray:
    """
    log p(Z_{k+1} | Z_k, state = i) for every (k, i).

    Returns log_B of shape (T - 1, K) where T = len(Z).
    """
    T = Z.shape[0]
    dim = Z.shape[1]
    K = params.K
    log2pi = dim * np.log(2.0 * np.pi)

    # Pre-factor each R_i = L_i L_i^T
    L = np.empty((K, dim, dim))
    logdet = np.empty(K)
    for i in range(K):
        Ri = params.R[i]
        try:
            L[i] = np.linalg.cholesky(Ri)
        except np.linalg.LinAlgError:
            # Tiny ridge to recover from a momentarily non-PD R
            L[i] = np.linalg.cholesky(Ri + 1e-10 * np.eye(dim))
        logdet[i] = 2.0 * np.sum(np.log(np.diag(L[i])))

    log_B = np.empty((T - 1, K))
    for i in range(K):
        dev = Z[:-1] - params.mu[i]                 # (T - 1, dim)
        pred = dev @ params.B[i].T                  # (T - 1, dim)
        res = Z[1:] - params.mu[i] - pred           # (T - 1, dim)
        # Mahalanobis: solve L z = res^T for z, then |z|^2 = res^T R^{-1} res
        z = solve_triangular(L[i], res.T, lower=True, check_finite=False)
        maha = np.sum(z * z, axis=0)                # (T - 1,)
        log_B[:, i] = -0.5 * (log2pi + logdet[i] + maha)
    return log_B


def _solve_lower_tri(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Kept for callers that prefer a single function name; identical to
    # scipy.linalg.solve_triangular(L, B, lower=True).
    return solve_triangular(L, B, lower=True, check_finite=False)


def _forward_backward(log_B: np.ndarray,
                      T_mat: np.ndarray,
                      pi: np.ndarray):
    """
    Scaled forward-backward.

    Returns
    -------
    nu : (T, K) posterior P(z_t = i | observations)
    xi_sum : (K, K) sum over t of P(z_t = i, z_{t+1} = j | observations)
    log_lik : float
    """
    Tm1, K = log_B.shape
    T = Tm1 + 1

    # Scale emissions to numerically reasonable range
    log_B_max = log_B.max(axis=1, keepdims=True)         # (T - 1, 1)
    B_scaled = np.exp(log_B - log_B_max)                 # (T - 1, K)

    # Forward (alpha) with row-scaling at every step
    alpha = np.empty((T, K))
    c = np.empty(T)                                      # scaling factors
    alpha[0] = pi
    c[0] = alpha[0].sum()
    if c[0] < 1e-300:
        c[0] = 1e-300
    alpha[0] /= c[0]

    for t in range(1, T):
        alpha[t] = B_scaled[t - 1] * (alpha[t - 1] @ T_mat)
        c[t] = alpha[t].sum()
        if c[t] < 1e-300:
            c[t] = 1e-300
        alpha[t] /= c[t]

    log_lik = float(np.sum(np.log(c)) + np.sum(log_B_max))

    # Backward (beta)
    beta = np.empty((T, K))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = T_mat @ (B_scaled[t] * beta[t + 1])
        beta[t] /= c[t + 1]

    nu = alpha * beta
    nu /= nu.sum(axis=1, keepdims=True) + 1e-300

    # Pairwise responsibilities, summed over time.  Vectorised form:
    #   xi_t[i, j] = alpha[t, i] * T[i, j] * B[t, j] * beta[t+1, j]
    # Sum over t with normalisation per t.
    #   shape (T-1, K, 1) * (K, K) * (T-1, 1, K) * (T-1, 1, K)
    a_t = alpha[:-1]                                      # (Tm1, K)
    b_tp1 = beta[1:]                                      # (Tm1, K)
    # First compute the joint factor (T-1, K, K)
    joint = (a_t[:, :, None] * T_mat[None, :, :]
             * (B_scaled[:, None, :] * b_tp1[:, None, :]))
    s = joint.sum(axis=(1, 2)) + 1e-300                   # (T-1,)
    xi_sum = (joint / s[:, None, None]).sum(axis=0)

    return nu, xi_sum, log_lik


def _viterbi(log_B: np.ndarray, T_mat: np.ndarray, pi: np.ndarray) -> np.ndarray:
    Tm1, K = log_B.shape
    T = Tm1 + 1
    log_T = np.log(np.maximum(T_mat, 1e-300))
    log_pi = np.log(np.maximum(pi, 1e-300))

    delta = np.empty((T, K))
    psi = np.empty((T, K), dtype=np.int64)
    delta[0] = log_pi
    for t in range(1, T):
        # delta[t, j] = max_i [ delta[t-1, i] + log_T[i, j] + log_B[t-1, j] ]
        scores = delta[t - 1][:, None] + log_T            # (K, K)
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = scores[psi[t], np.arange(K)] + log_B[t - 1]

    # Back-trace
    path = np.empty(T, dtype=np.int64)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


# ------------------------------------------------------------------
# M-step: literal Horenko Theorem 3.1
# ------------------------------------------------------------------
def _m_step_horenko(Z: np.ndarray, nu: np.ndarray, xi_sum: np.ndarray,
                    tau: float, ridge: float = 1e-10) -> HMMSDEParams:
    """
    Theorem 3.1 (paper page 742):

        Zbar^(i)  = sum_k nu_{k+1}(i) Z_k                    / W^(i)
        Cov^(i)   = sum_k nu_{k+1}(i) (Z_k - Zbar)(Z_k - Zbar)^T / W^(i)
        Cor^(i)   = ( sum_k nu_{k+1}(i) (Z_{k+1} - Zbar)(Z_k - Zbar)^T / W^(i) ) Cov^{-1}
        delta^(i) = sum_k nu_{k+1}(i) (Z_{k+1} - Z_k)        / W^(i)
        W^(i)     = sum_k nu_{k+1}(i)

    Then
        B^(i)  = exp(tau F^(i)) = Cor^(i)
        mu^(i) = Zbar^(i) - (I - Cor^(i))^{-1} delta^(i)
        R^(i)  = sum_k nu_{k+1}(i) d_k d_k^T / W^(i),
                 d_k = Z_{k+1} - mu^(i) - B^(i) (Z_k - mu^(i)).

    A tiny ridge is added to Cov before inversion only when needed for
    numerical robustness; the paper assumes Cov is positive definite.
    """
    T, dim = Z.shape
    K = nu.shape[1]
    Zk = Z[:-1]                              # (T - 1, dim) predictor
    Zp = Z[1:]                               # (T - 1, dim) response = Z_{k+1}
    w = nu[1:]                               # (T - 1, K) destination weights

    W = w.sum(axis=0) + 1e-15                # (K,)

    # Zbar (weighted mean over predictor side)
    Zbar = (w.T @ Zk) / W[:, None]           # (K, dim)

    # delta (weighted mean displacement)
    delta = (w.T @ (Zp - Zk)) / W[:, None]   # (K, dim)

    mu = np.empty((K, dim))
    B = np.empty((K, dim, dim))
    R = np.empty((K, dim, dim))

    for i in range(K):
        # Weighted cross-products centred at Zbar[i]
        dev_k = Zk - Zbar[i]                  # (T - 1, dim)
        dev_p = Zp - Zbar[i]                  # (T - 1, dim)
        wi = w[:, i][:, None]                 # (T - 1, 1)
        Wi = W[i]

        Cov = (wi * dev_k).T @ dev_k / Wi     # (dim, dim)
        Cov = 0.5 * (Cov + Cov.T)
        # Conservative ridge if Cov is degenerate
        eig_min = np.linalg.eigvalsh(Cov)[0]
        if eig_min < ridge:
            Cov = Cov + (ridge - eig_min) * np.eye(dim)

        Cor_num = (wi * dev_p).T @ dev_k / Wi  # (dim, dim), uses Z_{k+1}, not Zbar_+
        Cor = Cor_num @ np.linalg.inv(Cov)

        B[i] = Cor
        # mu^(i) from eq (29): mu = (I - B)^{-1} (Zbar_+ - B Zbar),
        # where Zbar_+ = sum nu Z_{k+1} / W.  Algebraically this is
        # mu = Zbar + (I - B)^{-1} delta.  Note that the restatement
        # of this formula inside Theorem 3.1 ("mu = Zbar - (I-Cor)^{-1} delta")
        # is a sign typo, invisible in stationary single-state tests
        # where delta vanishes, but material here.
        Zbar_plus = (wi * Zp).sum(axis=0) / Wi          # (dim,)
        rhs = Zbar_plus - B[i] @ Zbar[i]
        try:
            mu[i] = np.linalg.solve(np.eye(dim) - B[i], rhs)
        except np.linalg.LinAlgError:
            mu[i] = np.linalg.lstsq(np.eye(dim) - B[i], rhs, rcond=None)[0]

        # R^(i)(tau) = E_{nu_{k+1}}[ d_k d_k^T ] with d_k = Z_{k+1} - mu - B (Z_k - mu)
        d = Zp - mu[i] - (Zk - mu[i]) @ B[i].T   # (T - 1, dim)
        Ri = (wi * d).T @ d / Wi
        Ri = 0.5 * (Ri + Ri.T)
        # Floor any non-PD residual at a tiny eigenvalue
        eigvals, eigvecs = np.linalg.eigh(Ri)
        eigvals = np.maximum(eigvals, 1e-10)
        R[i] = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Transition matrix from xi-sums; row-stochastic
    T_mat = xi_sum.copy()
    T_mat = np.maximum(T_mat, 0.0)
    row = T_mat.sum(axis=1, keepdims=True) + 1e-15
    T_mat = T_mat / row

    # Initial distribution from nu[0]
    pi = nu[0] / (nu[0].sum() + 1e-15)

    return HMMSDEParams(K=K, dim=dim, tau=tau,
                        mu=mu, B=B, R=R, T=T_mat, pi=pi)


# ------------------------------------------------------------------
# Initialisation
# ------------------------------------------------------------------
def _init_kmeans(Z: np.ndarray, K: int, rng: np.random.Generator,
                 n_iter: int = 30) -> np.ndarray:
    """K-means++ initialisation, returns (K, dim) centres."""
    T, dim = Z.shape
    centres = np.empty((K, dim))
    # k-means++ seeding
    idx0 = int(rng.integers(T))
    centres[0] = Z[idx0]
    for k in range(1, K):
        d2 = np.min(np.sum((Z[:, None, :] - centres[None, :k, :]) ** 2, axis=2),
                    axis=1)
        # Sample proportional to d^2
        probs = d2 / (d2.sum() + 1e-15)
        idx = int(rng.choice(T, p=probs))
        centres[k] = Z[idx]
    # Lloyd iterations
    for _ in range(n_iter):
        d2 = np.sum((Z[:, None, :] - centres[None, :, :]) ** 2, axis=2)
        labels = np.argmin(d2, axis=1)
        new = centres.copy()
        for k in range(K):
            sel = labels == k
            if sel.sum() > 0:
                new[k] = Z[sel].mean(axis=0)
        if np.max(np.linalg.norm(new - centres, axis=1)) < 1e-8:
            centres = new
            break
        centres = new
    return centres


def _init_params(Z: np.ndarray, K: int, tau: float,
                 rng: np.random.Generator) -> HMMSDEParams:
    """
    Initialisation per Horenko step 1: zeroth iterates for the local SDE
    parameters lambda^(i) = (exp(tau F), Sigma Sigma^T, mu) and HMM params
    (T, pi).

    Recipe:
      - mu: K-means++ centres.
      - B: exp(-tau) * I (stable damped identity).
      - R: weighted residual covariance per cluster.
      - T: counts of K-means transitions (with pseudocount).
      - pi: cluster occupancy.
    """
    T_len, dim = Z.shape
    mu = _init_kmeans(Z, K, rng)

    d2 = np.sum((Z[:, None, :] - mu[None, :, :]) ** 2, axis=2)
    labels = np.argmin(d2, axis=1)

    B = np.stack([np.exp(-tau) * np.eye(dim) for _ in range(K)])

    # Initial R from per-cluster displacement covariance
    dZ = np.diff(Z, axis=0)
    R = np.empty((K, dim, dim))
    for k in range(K):
        sel = labels[:-1] == k
        if sel.sum() > max(20, dim + 2):
            cov_k = np.cov(dZ[sel].T) + 1e-6 * np.eye(dim)
        else:
            cov_k = 0.05 * np.eye(dim)
        # ensure PD
        eig, vec = np.linalg.eigh(0.5 * (cov_k + cov_k.T))
        eig = np.maximum(eig, 1e-6)
        R[k] = vec @ np.diag(eig) @ vec.T

    # Initial transition matrix from K-means label transitions
    Tmat = np.full((K, K), 1.0 / K)  # uniform fallback
    counts = np.zeros((K, K)) + 1.0  # pseudocount = 1
    for t in range(T_len - 1):
        counts[labels[t], labels[t + 1]] += 1.0
    counts /= counts.sum(axis=1, keepdims=True)
    Tmat = counts

    pi = np.bincount(labels, minlength=K).astype(float)
    pi = pi / (pi.sum() + 1e-15)
    # Avoid hard zeros
    pi = np.maximum(pi, 1e-3)
    pi /= pi.sum()

    return HMMSDEParams(K=K, dim=dim, tau=tau, mu=mu, B=B, R=R, T=Tmat, pi=pi)


# ------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------
def fit_hmmsde(Z: np.ndarray, tau: float, K: int,
               max_iter: int = 200, tol: float = 1e-5,
               n_init: int = 5, seed: Optional[int] = None,
               verbose: bool = False) -> HMMSDEResult:
    """
    Fit a K-state HMM-SDE by EM with multiple random restarts.

    Returns the best run by log-likelihood.
    """
    Z = np.asarray(Z, dtype=float)
    T_len, dim = Z.shape
    rng = np.random.default_rng(seed)

    n_params = K * (dim + dim * dim + dim * (dim + 1) // 2) \
        + K * (K - 1) + (K - 1)

    best: Optional[HMMSDEResult] = None
    best_ll = -np.inf

    for init_idx in range(n_init):
        params = _init_params(Z, K, tau, rng)
        ll_hist: List[float] = []
        converged = False
        prev_ll = -np.inf

        for it in range(max_iter):
            log_B = _emission_loglik(Z, params)
            nu, xi_sum, ll = _forward_backward(log_B, params.T, params.pi)
            ll_hist.append(ll)

            # Numerical guard: EM should never go down.  A tiny non-monotone
            # step (~1e-6 relative) is acceptable from floating-point noise.
            if it > 0 and ll < prev_ll - 1e-6 * abs(prev_ll) - 1e-8:
                warnings.warn(
                    f"[hmmsde] init={init_idx} iter={it}: log-likelihood "
                    f"decreased: {prev_ll:.6f} -> {ll:.6f}"
                )

            if it > 0 and abs(ll - prev_ll) < tol * max(1.0, abs(prev_ll)):
                converged = True
                break
            prev_ll = ll

            params = _m_step_horenko(Z, nu, xi_sum, tau)

        if verbose:
            print(f"  init {init_idx + 1}/{n_init}: ll={ll:.2f}, "
                  f"iters={it + 1}, converged={converged}")

        if ll > best_ll:
            best_ll = ll
            bic = -2.0 * ll + n_params * np.log(T_len)
            best = HMMSDEResult(
                params=params,
                nu=nu,
                log_likelihood=ll,
                log_likelihood_history=ll_hist,
                bic=bic,
                viterbi=_viterbi(log_B, params.T, params.pi),
                n_iter=it + 1,
                converged=converged,
            )

    assert best is not None
    return best

"""
Plots for the Horenko HMM-SDE fit on the Müller-Brown potential.

Five figures are produced:

  fig_01_potential.png    - heat-map of V with minima/saddles marked
  fig_02_trajectory.png   - trajectory overlaid on the potential
  fig_03_likelihood.png   - log-likelihood vs EM iteration (best init)
  fig_04_spectrum.png     - implied timescales of the K' state HMM
                            with the spectral gap that picks L_hat
  fig_05_results.png      - estimated mu_macro vs true minima; per-state
                            membership chi; estimated vs Kramers rates
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Tuple


def _potential_grid(potential, xlim=(-1.8, 1.3), ylim=(-0.6, 2.2), n=200):
    x = np.linspace(*xlim, n)
    y = np.linspace(*ylim, n)
    XX, YY = np.meshgrid(x, y)
    V = np.empty_like(XX)
    for i in range(n):
        for j in range(n):
            V[i, j] = potential.V(np.array([XX[i, j], YY[i, j]]))
    return XX, YY, V


def plot_potential(potential, path: str,
                   xlim=(-1.8, 1.3), ylim=(-0.6, 2.2)) -> None:
    XX, YY, V = _potential_grid(potential, xlim, ylim)
    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contourf(XX, YY, V, levels=25, cmap="viridis_r")
    ax.contour(XX, YY, V, levels=15, colors="k", linewidths=0.3, alpha=0.4)
    for i, m in enumerate(potential.minima):
        ax.plot(*m.position, "o", color="red", markersize=10,
                markeredgecolor="white")
        ax.annotate(f"min {i}", m.position, fontsize=9,
                    color="white", weight="bold",
                    xytext=(7, 5), textcoords="offset points")
    for i, s in enumerate(potential.saddles):
        ax.plot(*s.position, "x", color="orange", markersize=10, mew=2)
        ax.annotate(f"sad {i}", s.position, fontsize=9,
                    color="orange", weight="bold",
                    xytext=(7, 5), textcoords="offset points")
    fig.colorbar(cs, ax=ax, label="V(x)")
    ax.set_title(f"Müller-Brown potential (scale={potential.scale})")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_trajectory(potential, traj, viterbi, path: str,
                    n_subsample: int = 4000,
                    xlim=(-1.8, 1.3), ylim=(-0.6, 2.2)) -> None:
    XX, YY, V = _potential_grid(potential, xlim, ylim)
    K = int(viterbi.max()) + 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: trajectory coloured by time
    ax = axes[0]
    ax.contourf(XX, YY, V, levels=25, cmap="Greys", alpha=0.5)
    ax.contour(XX, YY, V, levels=12, colors="k", linewidths=0.3, alpha=0.3)
    sel = np.linspace(0, traj.X.shape[0] - 1, n_subsample).astype(int)
    sc = ax.scatter(traj.X[sel, 0], traj.X[sel, 1],
                    c=traj.t[sel], s=2, cmap="plasma")
    for m in potential.minima:
        ax.plot(*m.position, "o", color="red", markersize=8,
                markeredgecolor="white")
    plt.colorbar(sc, ax=ax, label="t")
    ax.set_title(f"Trajectory (T={traj.t[-1]:.0f}, ε={traj.eps}, τ={traj.tau})")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_xlim(xlim); ax.set_ylim(ylim)

    # Right: trajectory coloured by Viterbi state
    ax = axes[1]
    ax.contourf(XX, YY, V, levels=25, cmap="Greys", alpha=0.5)
    cmap = plt.cm.tab10
    for k in range(K):
        mask = viterbi == k
        sel_k = sel[mask[sel]]
        ax.scatter(traj.X[sel_k, 0], traj.X[sel_k, 1],
                   s=2, color=cmap(k), label=f"state {k}")
    ax.legend(loc="lower right", markerscale=3)
    ax.set_title("Viterbi-decoded hidden state")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_xlim(xlim); ax.set_ylim(ylim)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_likelihood(ll_history: List[float], path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ll_history, "o-", color="steelblue")
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("log-likelihood")
    ax.set_title(f"EM convergence ({len(ll_history)} iterations)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_spectrum(eigenvalues: np.ndarray, its: np.ndarray, L_hat: int,
                  tau: float, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.bar(np.arange(len(eigenvalues)), eigenvalues.real,
           color="steelblue", edgecolor="black")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7)
    ax.set_xlabel("index")
    ax.set_ylabel("eigenvalue (real)")
    ax.set_title("Eigenvalues of estimated $T$")

    ax = axes[1]
    its_plot = its.copy()
    finite = np.isfinite(its_plot)
    if np.any(finite):
        fill = max(float(its[finite].max()) * 1.5, 10.0 * tau)
    else:
        fill = 1.0
    its_plot[~finite] = fill
    ax.bar(np.arange(len(its)), its_plot, color="firebrick",
           edgecolor="black")
    ax.axvline(L_hat - 0.5, color="green", linestyle="--",
               linewidth=2, label=f"PCCA+ cut: $\\hat L$={L_hat}")
    ax.set_yscale("log")
    ax.set_xlabel("index")
    ax.set_ylabel("implied timescale $t_i = -\\tau / \\log|\\lambda_i|$")
    ax.set_title("Implied timescales")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_results(potential, agg_params, perm, chi,
                 nu_micro, Q_est, Q_kramers, path: str,
                 xlim=(-1.8, 1.3), ylim=(-0.6, 2.2)) -> None:
    """
    Three panels:
      - estimated mu_macro on top of potential, with true minima
      - membership chi (as a heatmap)
      - estimated vs Kramers off-diagonal rates (bar chart)
    """
    XX, YY, V = _potential_grid(potential, xlim, ylim)
    K = agg_params.K
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Panel 1: estimated and true centres
    ax = axes[0]
    ax.contourf(XX, YY, V, levels=25, cmap="Greys", alpha=0.6)
    ax.contour(XX, YY, V, levels=12, colors="k", linewidths=0.3, alpha=0.3)
    cmap = plt.cm.tab10
    for j, m in enumerate(potential.minima):
        ax.plot(*m.position, "o", color="red", markersize=14,
                markeredgecolor="white", label="true min" if j == 0 else None)
    for i in range(K):
        c = cmap(i)
        ax.plot(*agg_params.mu[i], "s", color=c, markersize=12,
                markeredgecolor="black",
                label=f"est. state {i} → true min {perm[i]}")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Estimated macrostate centres")

    # Panel 2: chi heatmap
    #ax = axes[1]
    #im = ax.imshow(chi, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    #ax.set_xlabel("macro index")
    #ax.set_ylabel("micro index")
    #ax.set_xticks(np.arange(chi.shape[1]))
    #ax.set_yticks(np.arange(chi.shape[0]))
    #for i in range(chi.shape[0]):
    #    for j in range(chi.shape[1]):
    #        ax.text(j, i, f"{chi[i, j]:.2f}", ha="center", va="center",
    #                color="white" if chi[i, j] < 0.5 else "black",
    #                fontsize=8)
    #plt.colorbar(im, ax=ax, label="χ")
    #ax.set_title("PCCA+ membership matrix")

    # Panel 3: rate comparison
    ax = axes[1]
    K_true = Q_kramers.shape[0]
    pairs = [(i, j) for i in range(K_true) for j in range(K_true) if i != j]
    xticklabels = [f"{i}→{j}" for (i, j) in pairs]
    est_rates = [Q_est[i, j] for (i, j) in pairs]
    kramers_rates = [Q_kramers[i, j] for (i, j) in pairs]
    xs = np.arange(len(pairs))
    width = 0.35
    ax.bar(xs - width / 2, kramers_rates, width, label="Kramers",
           color="firebrick", edgecolor="black")
    ax.bar(xs + width / 2, est_rates, width, label="estimated",
           color="steelblue", edgecolor="black")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xticks(xs)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("rate $k_{i \\to j}$")
    ax.set_title("Kramers vs estimated jump rates")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

"""
Driver for the Horenko HMM-SDE experiment on the scaled Müller-Brown
potential at eps = 0.15.

Workflow:

  1. Generate an overdamped Langevin trajectory at eps = 0.25 in the
     potential V_MB scaled by 0.01, using Euler-Maruyama with a small
     integration step dt_sim = 5e-3 and thinning thin = 20 so that the
     observation lag tau = 0.1.

  2. Fit an HMM-SDE with K' overestimated hidden states (K' = 6 by
     default), using EM with literal Theorem 3.1 M-step formulas.
     Multi-start: n_init = 5; keep the run with the largest log-
     likelihood at convergence.

  3. Run PCCA+ on the estimated K' x K' transition matrix.  Pick the
     macrostate count L_hat from the leading implied-timescale gap.
     Aggregate the K' per-microstate emissions into L_hat per-macrostate
     emissions using the chi-weighted Galerkin formulas.

  4. Diagnostics:
       - Symmetry of F = log(B)/tau per estimated macrostate
         (overdamped Langevin -> symmetric F).
       - Sigma Sigma^T from R via the asymptotic form
         -(F Cov + Cov F^T); compare with the FDR prediction 2 eps I.
       - Hungarian matching of estimated mu_macro to the true MB minima.
       - Compare the estimated continuous-time generator Q = log(T)/tau
         to the Eyring-Kramers ground truth.

All numeric results are printed to stdout and saved to
results/results.json; figures go to figures/ .
"""

import argparse
import json
import os
import time
import numpy as np

from horenko_hmmsde.potential import mueller_brown, kramers_rate_matrix
from horenko_hmmsde.langevin import simulate_overdamped, assign_well
from horenko_hmmsde.hmmsde import fit_hmmsde
from horenko_hmmsde.pcca import (sort_eigendecomposition, implied_timescales,
                                  select_macrostate_count, pcca_plus,
                                  aggregate_params)
from horenko_hmmsde.diagnostics import (principal_log_F, symmetry_defect,
                                        sigma_sigmaT_from_R, well_assignment,
                                        generator_from_T, match_generator)
from horenko_hmmsde.plotting import (plot_potential, plot_trajectory,
                                     plot_likelihood, plot_spectrum,
                                     plot_results)


def _np2py(x):
    """Recursive numpy -> Python for JSON."""
    if isinstance(x, dict):
        return {k: _np2py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_np2py(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eps", type=float, default=0.15,
                   help="noise intensity in dX = -grad V dt + sqrt(2 eps) dW")
    p.add_argument("--scale", type=float, default=0.01,
                   help="scale factor multiplying the standard MB potential")
    p.add_argument("--T-phys", type=float, default=2000.0,
                   help="total simulated physical time")
    p.add_argument("--dt-sim", type=float, default=5e-3,
                   help="Euler-Maruyama integrator step")
    p.add_argument("--thin", type=int, default=20,
                   help="store every thin-th sample (tau = dt_sim*thin)")
    p.add_argument("--K-prime", type=int, default=6,
                   help="number of hidden states for the over-estimated HMM")
    p.add_argument("--n-init", type=int, default=5,
                   help="number of EM restarts")
    p.add_argument("--max-iter", type=int, default=200,
                   help="max EM iterations per restart")
    p.add_argument("--tol", type=float, default=1e-5,
                   help="EM convergence tolerance on log-likelihood")
    p.add_argument("--gap-threshold", type=float, default=2.0,
                   help="minimum implied-timescale ratio for the PCCA+ cut")
    p.add_argument("--min-K", type=int, default=2,
                   help="minimum macrostate count")
    p.add_argument("--L-hat", type=int, default=None,
                   help="override the macrostate count (skip gap detection)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed (controls both trajectory and EM init)")
    p.add_argument("--out-dir", type=str, default=".",
                   help="output directory")
    p.add_argument("--no-plots", action="store_true",
                   help="skip plot generation")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    fig_dir = os.path.join(args.out_dir, "figures")
    res_dir = os.path.join(args.out_dir, "results")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    log = (lambda *a, **k: None) if args.quiet else print
    summary = {"args": vars(args)}

    # ── 1. Potential and ground-truth Kramers rates ─────────────────
    log("=" * 70)
    log("STEP 1: construct the scaled Müller-Brown potential")
    log("=" * 70)
    potential = mueller_brown(scale=args.scale)
    log(f"  minima:   {[f'V={m.energy:.4f}' for m in potential.minima]}")
    log(f"  saddles:  {[f'V={s.energy:.4f}' for s in potential.saddles]}")
    Q_kramers, _ = kramers_rate_matrix(potential, eps=args.eps)
    log(f"  Kramers Q at eps={args.eps}:")
    for row in np.round(Q_kramers, 5):
        log(f"    {row}")
    summary["potential"] = {
        "minima": [{"position": m.position.tolist(), "V": m.energy,
                    "hess_eigs": m.hessian_eigenvalues.tolist()}
                   for m in potential.minima],
        "saddles": [{"position": s.position.tolist(), "V": s.energy,
                     "hess_eigs": s.hessian_eigenvalues.tolist()}
                    for s in potential.saddles],
        "kramers_Q": Q_kramers.tolist(),
    }

    # ── 2. Simulate the trajectory ──────────────────────────────────
    log("=" * 70)
    log("STEP 2: simulate overdamped Langevin trajectory")
    log("=" * 70)
    t0 = time.time()
    traj = simulate_overdamped(
        potential, eps=args.eps, T_phys=args.T_phys,
        dt_sim=args.dt_sim, thin=args.thin, seed=args.seed,
    )
    sim_time = time.time() - t0
    log(f"  T_phys = {args.T_phys}, dt_sim = {args.dt_sim}, thin = {args.thin}")
    log(f"  -> {traj.X.shape[0]} samples at tau = {traj.tau}")
    log(f"  -> wall time: {sim_time:.2f}s")

    true_labels = assign_well(traj.X, potential)
    well_counts = np.bincount(true_labels, minlength=len(potential.minima))
    log(f"  hard well occupancy (nearest minimum): {well_counts.tolist()}")
    summary["trajectory"] = {
        "n_samples": int(traj.X.shape[0]),
        "tau": traj.tau,
        "wall_time_s": sim_time,
        "true_well_counts": well_counts.tolist(),
    }

    # ── 3. Fit K'-state HMM-SDE ─────────────────────────────────────
    log("=" * 70)
    log(f"STEP 3: fit HMM-SDE with K' = {args.K_prime} (over-estimate)")
    log("=" * 70)
    t0 = time.time()
    fit = fit_hmmsde(
        traj.X, tau=traj.tau, K=args.K_prime,
        max_iter=args.max_iter, tol=args.tol,
        n_init=args.n_init, seed=args.seed, verbose=not args.quiet,
    )
    fit_time = time.time() - t0
    log(f"  best log-likelihood = {fit.log_likelihood:.3f} "
        f"(BIC = {fit.bic:.1f})")
    log(f"  converged in {fit.n_iter} EM iterations")
    log(f"  EM wall time: {fit_time:.1f}s")
    log("  estimated mu (sorted by stationary occupancy):")
    occ = fit.nu.mean(axis=0)
    order = np.argsort(-occ)
    for i in order:
        log(f"    state {i}: mu = {np.round(fit.params.mu[i], 4)}, "
            f"occ = {occ[i]:.4f}")
    summary["fit"] = {
        "K_prime": args.K_prime,
        "log_likelihood": fit.log_likelihood,
        "bic": fit.bic,
        "n_iter": fit.n_iter,
        "converged": fit.converged,
        "wall_time_s": fit_time,
        "mu": fit.params.mu.tolist(),
        "B": fit.params.B.tolist(),
        "R": fit.params.R.tolist(),
        "T": fit.params.T.tolist(),
        "pi": fit.params.pi.tolist(),
        "stationary_occ": occ.tolist(),
    }

    # ── 4. PCCA+ reduction ──────────────────────────────────────────
    log("=" * 70)
    log("STEP 4: PCCA+ reduction to metastable macrostates")
    log("=" * 70)
    eigs, _ = sort_eigendecomposition(fit.params.T)
    its = implied_timescales(eigs, traj.tau)
    log(f"  T eigenvalues (sorted desc): "
        f"{np.round(eigs, 5).tolist()}")
    log(f"  implied timescales:          "
        f"{np.round(its, 3).tolist()}")
    if args.L_hat is not None:
        L_hat = args.L_hat
        log(f"  override: L_hat = {L_hat}")
    else:
        L_hat, _ = select_macrostate_count(
            eigs, traj.tau, min_K=args.min_K,
            gap_threshold=args.gap_threshold, verbose=not args.quiet,
        )
        log(f"  -> L_hat from spectral gap (threshold {args.gap_threshold}): {L_hat}")
    chi, _, vertices = pcca_plus(fit.params.T, L_hat)
    log(f"  vertex micro indices: {vertices}")
    log(f"  chi (rows = microstates, cols = macrostates):")
    for row in np.round(chi, 3):
        log(f"    {row.tolist()}")
    agg = aggregate_params(fit.params, chi, fit.nu, fit.params.T,
                            Z=traj.X, tau=traj.tau)
    log(f"  aggregated mu_macro:")
    for i in range(L_hat):
        log(f"    macro {i}: mu = {np.round(agg.mu[i], 4)}")
    summary["pcca"] = {
        "eigenvalues": eigs.tolist(),
        "implied_timescales": its.tolist(),
        "L_hat": int(L_hat),
        "vertices": list(vertices),
        "chi": chi.tolist(),
        "mu_macro": agg.mu.tolist(),
        "T_macro": agg.T.tolist(),
        "B_macro": agg.B.tolist(),
        "R_macro": agg.R.tolist(),
        "pi_macro": agg.pi.tolist(),
    }

    # ── 5. Diagnostics: F symmetry & Sigma Sigma^T per macrostate ───
    log("=" * 70)
    log("STEP 5: post-fit diagnostics")
    log("=" * 70)
    # Recompute Cov per macrostate via macro-membership-weighted posterior.
    # nu_macro = nu @ chi
    nu_macro = np.clip(fit.nu @ chi, 0, 1)
    nu_macro = nu_macro / (nu_macro.sum(axis=1, keepdims=True) + 1e-15)

    Zk = traj.X[:-1]
    diag_per_state = []
    for i in range(L_hat):
        F_i = principal_log_F(agg.B[i], agg.tau)
        sym_def = symmetry_defect(F_i)
        # Macro Cov from macro-membership-weighted statistics
        w = nu_macro[1:, i]
        W = w.sum() + 1e-15
        Zbar = (w[:, None] * Zk).sum(axis=0) / W
        Cov = ((w[:, None] * (Zk - Zbar)).T @ (Zk - Zbar)) / W
        Cov = 0.5 * (Cov + Cov.T)
        SST = sigma_sigmaT_from_R(agg.B[i], agg.R[i], F_i, Cov)
        SST = 0.5 * (SST + SST.T)
        # FDR check: should be ≈ 2 eps I
        FDR_target = 2.0 * args.eps
        FDR_diag_err = np.abs(np.diag(SST) - FDR_target) / FDR_target
        FDR_offdiag = abs(SST[0, 1]) / FDR_target
        log(f"  macrostate {i} (mu={np.round(agg.mu[i], 3).tolist()}):")
        log(f"    F = log(B)/tau =")
        for row in np.round(F_i, 4):
            log(f"      {row.tolist()}")
        log(f"    symmetry defect ||F-F^T||/||F+F^T|| = {sym_def:.3e}")
        log(f"    Sigma Sigma^T  =")
        for row in np.round(SST, 4):
            log(f"      {row.tolist()}")
        log(f"    FDR prediction (2 eps I) = "
            f"diag={FDR_target:.4f}, offdiag=0")
        log(f"    FDR diag rel error = "
            f"[{FDR_diag_err[0]:.3f}, {FDR_diag_err[1]:.3f}]; "
            f"|offdiag|/diag = {FDR_offdiag:.3f}")
        diag_per_state.append({
            "F": F_i.tolist(),
            "symmetry_defect": sym_def,
            "Sigma_SigmaT": SST.tolist(),
            "FDR_diag_rel_err": FDR_diag_err.tolist(),
            "FDR_offdiag_over_target": float(FDR_offdiag),
        })

    # Hungarian matching
    true_min = np.stack([m.position for m in potential.minima])
    perm, match_dist = well_assignment(agg.mu, true_min)
    log(f"  well assignment (est → true): perm = {perm.tolist()}, "
        f"distances = {np.round(match_dist, 3).tolist()}")

    # Estimated generator and comparison
    Q_est_raw = generator_from_T(agg.T, agg.tau)
    Q_match = match_generator(Q_est_raw, perm)
    log(f"  Q_kramers:")
    for row in np.round(Q_kramers, 4):
        log(f"    {row.tolist()}")
    log(f"  Q_estimated (ordered by matched true-minimum labels):")
    for row in np.round(Q_match, 4):
        log(f"    {row.tolist()}")
    M = Q_kramers.shape[0]
    rate_err = []
    if Q_match.shape == Q_kramers.shape and set(perm.tolist()) == set(range(M)):
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                qk = Q_kramers[i, j]
                qe = Q_match[i, j]
                if qk > 0:
                    ratio = qe / qk
                    log(f"    rate {i}->{j}: kramers={qk:.4f}, "
                        f"estimated={qe:.4f}, ratio est/kramers = {ratio:.3f}")
                    rate_err.append({"i": i, "j": j, "kramers": float(qk),
                                     "estimated": float(qe), "ratio": float(ratio)})
                else:
                    log(f"    rate {i}->{j}: kramers=0 (no shared saddle), "
                        f"estimated={qe:.4f}")
                    rate_err.append({"i": i, "j": j, "kramers": float(qk),
                                     "estimated": float(qe), "ratio": None})
    else:
        log("  Skipping pairwise Kramers rate comparison because PCCA+ did "
            "not return a one-to-one set of three macrostates.")

    summary["diagnostics"] = {
        "per_state": diag_per_state,
        "perm_est_to_true": perm.tolist(),
        "match_dist": match_dist.tolist(),
        "Q_kramers": Q_kramers.tolist(),
        "Q_estimated_raw": Q_est_raw.tolist(),
        "Q_estimated_matched": Q_match.tolist(),
        "rate_comparison": rate_err,
    }

    # ── 6. Plots ────────────────────────────────────────────────────
    if not args.no_plots:
        log("=" * 70)
        log("STEP 6: figures")
        log("=" * 70)
        plot_potential(potential, os.path.join(fig_dir, "fig_01_potential.png"))
        plot_trajectory(potential, traj, fit.viterbi,
                        os.path.join(fig_dir, "fig_02_trajectory.png"))
        plot_likelihood(fit.log_likelihood_history,
                        os.path.join(fig_dir, "fig_03_likelihood.png"))
        plot_spectrum(eigs, its, L_hat, traj.tau,
                      os.path.join(fig_dir, "fig_04_spectrum.png"))
        plot_results(potential, agg, perm, chi, fit.nu, Q_match, Q_kramers,
                     os.path.join(fig_dir, "fig_05_results.png"))
        log(f"  figures written to {fig_dir}/")

    # ── 7. Persist results ──────────────────────────────────────────
    out_json = os.path.join(res_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(_np2py(summary), f, indent=2)
    log(f"  results written to {out_json}")
    log("=" * 70)
    log("DONE")
    log("=" * 70)


if __name__ == "__main__":
    # python run_experiment.py   --eps 0.15   --T-phys 3000   --K-prime 5   --n-init 5   --max-iter 200   --tol 1e-6   --seed 42
    main()

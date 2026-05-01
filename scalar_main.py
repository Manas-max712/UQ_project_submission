from __future__ import annotations
import os
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sstats
import pickle
import time
import warnings
import argparse
warnings.filterwarnings("ignore")

from monte_carlo import InputSpec, run_monte_carlo
from pce import build_pce, sample_surrogate


def run_mc(nx, N, spec, seed=2026):

    print(f"\n{'#' * 70}")
    print(f"# SCALAR MC  --  RVE {nx}x{nx}  --  N = {N}")
    print(f"{'#' * 70}", flush=True)
    t0 = time.time()
    res = run_monte_carlo(N=N, spec=spec, nx=nx, ny=nx,
                          seed=seed, verbose=True)
    elapsed = time.time() - t0
    print(f"\n  scalar MC {nx}x{nx} took {elapsed:.1f}s", flush=True)
    return {
        "kind": "MC", "nx": nx, "ny": nx, "N": N, "n_dim": 4,
        "E_star": res.E_star, "nu_star": res.nu_star,
        "elapsed_sec": elapsed,
        "mean_E":  float(np.nanmean(res.E_star)),
        "var_E":   float(np.nanvar(res.E_star, ddof=1)),
        "mean_nu": float(np.nanmean(res.nu_star)),
        "var_nu":  float(np.nanvar(res.nu_star, ddof=1)),
    }


def run_pce(nx, order, quad_order, spec, sparse=True, quad_cache=None,
            pdf_samples=100000):

    print(f"\n{'#' * 70}")
    print(f"# SCALAR PCE  --  RVE {nx}x{nx}  --  p = {order}  --  Q = {quad_order}")
    print(f"{'#' * 70}", flush=True)
    t0 = time.time()
    try:
        cache_key = (nx, quad_order, sparse)
        precomputed = None if quad_cache is None else quad_cache.get(cache_key)
        res = build_pce(order, spec, nx=nx, ny=nx,
                        quadrature_order=quad_order,
                        precomputed_quad=precomputed,
                        sparse=sparse, verbose=True)
        if quad_cache is not None and precomputed is None:
            quad_cache[cache_key] = (
                res.xi_grid, res.weights, res.E_star_model, res.nu_star_model
            )
        elapsed = time.time() - t0
        print(f"\n  scalar PCE {nx}x{nx} p={order}, Q={quad_order} took {elapsed:.1f}s",
              flush=True)
        coeffs_E_arr  = (np.asarray(res.coeffs_E).flatten()
                         if hasattr(res, "coeffs_E")
                         and res.coeffs_E is not None else None)
        coeffs_nu_arr = (np.asarray(res.coeffs_nu).flatten()
                         if hasattr(res, "coeffs_nu")
                         and res.coeffs_nu is not None else None)
        E_pdf_samples = None
        nu_pdf_samples = None
        if pdf_samples and pdf_samples > 0:
            E_s, nu_s = sample_surrogate(
                res,
                N=pdf_samples,
                seed=2026 + 1000 * nx + 100 * order + quad_order,
            )
            E_pdf_samples = np.asarray(E_s, dtype=np.float32)
            nu_pdf_samples = np.asarray(nu_s, dtype=np.float32)
        return {
            "kind": "PCE", "nx": nx, "ny": nx, "order": order,
            "quadrature_order": quad_order, "n_dim": 4,
            "n_basis": getattr(res, "n_basis", None),
            "n_quad":  getattr(res, "n_quad",  None),
            "coeffs_E":  coeffs_E_arr,
            "coeffs_nu": coeffs_nu_arr,
            "E_pdf_samples": E_pdf_samples,
            "nu_pdf_samples": nu_pdf_samples,
            "n_pdf_samples": int(pdf_samples or 0),
            "mean_E":  float(res.mean_E),
            "var_E":   float(res.var_E),
            "mean_nu": float(res.mean_nu),
            "var_nu":  float(res.var_nu),
            "elapsed_sec": elapsed,
        }
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  scalar PCE {nx}x{nx} p={order}, Q={quad_order} FAILED: {e}",
              flush=True)
        return {
            "kind": "PCE_failed", "nx": nx, "order": order,
            "quadrature_order": quad_order, "n_dim": 4,
            "error": str(e), "elapsed_sec": elapsed,
        }


def make_plots(results, outdir, spec=None):
    os.makedirs(outdir, exist_ok=True)
    if spec is None:
        spec = InputSpec()


    fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.5))
    theta_x = np.linspace(spec.theta_mean_deg - 4 * spec.theta_std_deg,
                          spec.theta_mean_deg + 4 * spec.theta_std_deg, 400)
    axs[0, 0].plot(theta_x,
                   sstats.norm.pdf(theta_x, loc=spec.theta_mean_deg,
                                   scale=spec.theta_std_deg),
                   lw=1.8)
    axs[0, 0].set_title("Input density: θ")
    axs[0, 0].set_xlabel("θ [deg]")
    axs[0, 0].set_ylabel("density")

    for ax, mean, cov, label in [
        (axs[0, 1], spec.t_mean, spec.t_cov, "t"),
        (axs[1, 0], spec.l_mean, spec.l_cov, "l"),
        (axs[1, 1], spec.E_mean, spec.E_cov, "E"),
    ]:
        mu, sigma = spec.lognormal_params(mean, cov)
        dist = sstats.lognorm(s=sigma, scale=np.exp(mu))
        x = np.linspace(dist.ppf(0.001), dist.ppf(0.999), 400)
        ax.plot(x, dist.pdf(x), lw=1.8)
        ax.set_title(f"Input density: {label}")
        ax.set_xlabel(label)
        ax.set_ylabel("density")

    for ax in axs.flat:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fname = "scalar_input_distributions.png"
    fig.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")

    mc_jobs  = sorted([r for r in results if r.get("kind") == "MC"],
                      key=lambda r: r["nx"])
    pce_jobs = sorted([r for r in results if r.get("kind") == "PCE"],
                      key=lambda r: (r["nx"], r["order"],
                                     r.get("quadrature_order", r["order"])))

    if not mc_jobs:
        print("  no MC jobs — skipping MC-dependent plots")
    if not pce_jobs:
        print("  no PCE jobs — skipping PCE-dependent plots")


    for mc in mc_jobs:
        nx = mc["nx"]

        def running_moments(values, max_points=700):
            values = values[np.isfinite(values)]
            n_total = len(values)
            if n_total == 0:
                return np.array([]), np.array([]), np.array([])
            stride = max(1, n_total // max_points)
            idx = np.arange(1, n_total + 1)
            keep = np.r_[np.arange(stride - 1, n_total, stride), n_total - 1]
            keep = np.unique(keep)
            run_mean = np.cumsum(values) / idx
            run_second = np.cumsum(values ** 2) / idx
            return idx[keep], run_mean[keep], run_second[keep]

        n_E, mean_E, second_E = running_moments(mc["E_star"])
        n_nu, mean_nu, second_nu = running_moments(mc["nu_star"])
        if len(n_E) == 0 or len(n_nu) == 0:
            print(f"  no finite MC samples for {nx}x{nx} — skipping MC convergence plot")
            continue

        fig, axs = plt.subplots(2, 2, figsize=(12, 8.2), sharex=False)
        axs[0, 0].plot(n_E, mean_E, color="C0", lw=1.5)
        axs[0, 0].axhline(mean_E[-1], color="k", ls="--", lw=1.0,
                          label=f"final N={n_E[-1]:,}")
        axs[0, 0].set_title("Running mean of E*")
        axs[0, 0].set_xlabel("MC samples")
        axs[0, 0].set_ylabel("E[E*]")

        axs[0, 1].plot(n_E, second_E, color="C1", lw=1.5)
        axs[0, 1].axhline(second_E[-1], color="k", ls="--", lw=1.0,
                          label=f"final N={n_E[-1]:,}")
        axs[0, 1].set_title("Running second moment of E*")
        axs[0, 1].set_xlabel("MC samples")
        axs[0, 1].set_ylabel("E[(E*)²]")

        axs[1, 0].plot(n_nu, mean_nu, color="C0", lw=1.5)
        axs[1, 0].axhline(mean_nu[-1], color="k", ls="--", lw=1.0,
                          label=f"final N={n_nu[-1]:,}")
        axs[1, 0].set_title("Running mean of ν*")
        axs[1, 0].set_xlabel("MC samples")
        axs[1, 0].set_ylabel("E[ν*]")

        axs[1, 1].plot(n_nu, second_nu, color="C1", lw=1.5)
        axs[1, 1].axhline(second_nu[-1], color="k", ls="--", lw=1.0,
                          label=f"final N={n_nu[-1]:,}")
        axs[1, 1].set_title("Running second moment of ν*")
        axs[1, 1].set_xlabel("MC samples")
        axs[1, 1].set_ylabel("E[(ν*)²]")

        for ax in axs.flat:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(f"Monte Carlo convergence at {nx}×{nx} RVE", y=1.01)
        fig.tight_layout()
        fname = f"scalar_mc_convergence_RVE_{nx}x{nx}.png"
        fig.savefig(os.path.join(outdir, fname), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {fname}")

    if not pce_jobs:
        return

    def q_of(row):
        return row.get("quadrature_order", row["order"])

    def moment(row, key):
        if key == "mean_E":
            return row["mean_E"]
        if key == "second_E":
            if row.get("kind") == "MC" and "E_star" in row:
                vals = row["E_star"][np.isfinite(row["E_star"])]
                return float(np.mean(vals ** 2))
            return row["var_E"] + row["mean_E"] ** 2
        if key == "mean_nu":
            return row["mean_nu"]
        if key == "second_nu":
            if row.get("kind") == "MC" and "nu_star" in row:
                vals = row["nu_star"][np.isfinite(row["nu_star"])]
                return float(np.mean(vals ** 2))
            return row["var_nu"] + row["mean_nu"] ** 2
        raise KeyError(key)

    def smoothed_density(values, x_grid, bins=180):
        values = np.asarray(values)
        values = values[np.isfinite(values)]
        if len(values) < 2:
            return np.full_like(x_grid, np.nan, dtype=float)
        if np.nanmax(values) <= np.nanmin(values):
            return np.full_like(x_grid, np.nan, dtype=float)
        hist, edges = np.histogram(values, bins=bins, range=(x_grid[0], x_grid[-1]),
                                   density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        kernel = np.array([1, 4, 7, 10, 7, 4, 1], dtype=float)
        kernel /= kernel.sum()
        hist = np.convolve(hist, kernel, mode="same")
        return np.interp(x_grid, centers, hist, left=0.0, right=0.0)

    moment_specs = [
        ("mean_E", "E[E*]", "Mean of E*"),
        ("second_E", "E[(E*)²]", "Second moment of E*"),
        ("mean_nu", "E[ν*]", "Mean of ν*"),
        ("second_nu", "E[(ν*)²]", "Second moment of ν*"),
    ]
    rve_sizes = sorted({r["nx"] for r in pce_jobs})

    for nx in rve_sizes:
        nx_jobs = [r for r in pce_jobs if r["nx"] == nx]
        nx_mc = next((r for r in mc_jobs if r["nx"] == nx), None)
        orders = sorted({r["order"] for r in nx_jobs})
        quad_orders = sorted({q_of(r) for r in nx_jobs})


        if nx_mc is not None:
            E_mc = nx_mc["E_star"][np.isfinite(nx_mc["E_star"])]
            nu_mc = nx_mc["nu_star"][np.isfinite(nx_mc["nu_star"])]
            for p in orders:
                pdf_jobs = sorted([r for r in nx_jobs if r["order"] == p],
                                  key=q_of)
                pdf_jobs = [
                    r for r in pdf_jobs
                    if r.get("E_pdf_samples") is not None
                    and r.get("nu_pdf_samples") is not None
                ]
                if len(E_mc) <= 1 or len(nu_mc) <= 1 or not pdf_jobs:
                    continue

                fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))

                E_all = [E_mc] + [
                    np.asarray(r["E_pdf_samples"])
                    for r in pdf_jobs
                ]
                lo, hi = np.percentile(np.concatenate(E_all), [0.5, 99.5])
                pad = 0.08 * (hi - lo) if hi > lo else 1.0
                x = np.linspace(lo - pad, hi + pad, 500)
                axs[0].plot(
                    x, smoothed_density(E_mc, x), color="red", lw=2.0,
                    label=f"Monte Carlo (N={len(E_mc):,})",
                )
                for row in pdf_jobs:
                    axs[0].plot(
                        x,
                        smoothed_density(row["E_pdf_samples"], x),
                        lw=1.3,
                        label=f"Q={q_of(row)}",
                    )
                axs[0].set_title("Density convergence for E*")
                axs[0].set_xlabel("E*")
                axs[0].set_ylabel("density")
                axs[0].grid(alpha=0.3)
                axs[0].legend(fontsize=8)

                nu_all = [nu_mc] + [
                    np.asarray(r["nu_pdf_samples"])
                    for r in pdf_jobs
                ]
                lo, hi = np.percentile(np.concatenate(nu_all), [0.5, 99.5])
                pad = 0.08 * (hi - lo) if hi > lo else 1.0
                x = np.linspace(lo - pad, hi + pad, 500)
                axs[1].plot(
                    x, smoothed_density(nu_mc, x), color="red", lw=2.0,
                    label=f"Monte Carlo (N={len(nu_mc):,})",
                )
                for row in pdf_jobs:
                    axs[1].plot(
                        x,
                        smoothed_density(row["nu_pdf_samples"], x),
                        lw=1.3,
                        label=f"Q={q_of(row)}",
                    )
                axs[1].set_title("Density convergence for ν*")
                axs[1].set_xlabel("ν*")
                axs[1].set_ylabel("density")
                axs[1].grid(alpha=0.3)
                axs[1].legend(fontsize=8)

                fig.suptitle(
                    f"Density convergence with Q, p={p}, {nx}×{nx} RVE",
                    y=1.02,
                )
                fig.tight_layout()
                fname = f"scalar_density_convergence_Q_RVE_{nx}x{nx}_p{p}.png"
                fig.savefig(os.path.join(outdir, fname), dpi=150,
                            bbox_inches="tight")
                plt.close(fig)
                print(f"  saved {fname}")


            for q in quad_orders:
                pdf_jobs = sorted([r for r in nx_jobs if q_of(r) == q],
                                  key=lambda row: row["order"])
                pdf_jobs = [
                    r for r in pdf_jobs
                    if r.get("E_pdf_samples") is not None
                    and r.get("nu_pdf_samples") is not None
                ]
                if len(E_mc) <= 1 or len(nu_mc) <= 1 or not pdf_jobs:
                    continue

                fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))

                E_all = [E_mc] + [
                    np.asarray(r["E_pdf_samples"])
                    for r in pdf_jobs
                ]
                lo, hi = np.percentile(np.concatenate(E_all), [0.5, 99.5])
                pad = 0.08 * (hi - lo) if hi > lo else 1.0
                x = np.linspace(lo - pad, hi + pad, 500)
                axs[0].plot(
                    x, smoothed_density(E_mc, x), color="red", lw=2.0,
                    label=f"Monte Carlo (N={len(E_mc):,})",
                )
                for row in pdf_jobs:
                    axs[0].plot(
                        x,
                        smoothed_density(row["E_pdf_samples"], x),
                        lw=1.3,
                        label=f"p={row['order']}",
                    )
                axs[0].set_title("Density convergence for E*")
                axs[0].set_xlabel("E*")
                axs[0].set_ylabel("density")
                axs[0].grid(alpha=0.3)
                axs[0].legend(fontsize=8)

                nu_all = [nu_mc] + [
                    np.asarray(r["nu_pdf_samples"])
                    for r in pdf_jobs
                ]
                lo, hi = np.percentile(np.concatenate(nu_all), [0.5, 99.5])
                pad = 0.08 * (hi - lo) if hi > lo else 1.0
                x = np.linspace(lo - pad, hi + pad, 500)
                axs[1].plot(
                    x, smoothed_density(nu_mc, x), color="red", lw=2.0,
                    label=f"Monte Carlo (N={len(nu_mc):,})",
                )
                for row in pdf_jobs:
                    axs[1].plot(
                        x,
                        smoothed_density(row["nu_pdf_samples"], x),
                        lw=1.3,
                        label=f"p={row['order']}",
                    )
                axs[1].set_title("Density convergence for ν*")
                axs[1].set_xlabel("ν*")
                axs[1].set_ylabel("density")
                axs[1].grid(alpha=0.3)
                axs[1].legend(fontsize=8)

                fig.suptitle(
                    f"Density convergence with p, Q={q}, {nx}×{nx} RVE",
                    y=1.02,
                )
                fig.tight_layout()
                fname = f"scalar_density_convergence_p_RVE_{nx}x{nx}_Q{q}.png"
                fig.savefig(os.path.join(outdir, fname), dpi=150,
                            bbox_inches="tight")
                plt.close(fig)
                print(f"  saved {fname}")


        best_q = max(quad_orders)
        coeff_jobs = sorted([r for r in nx_jobs if q_of(r) == best_q],
                            key=lambda row: row["order"])
        if coeff_jobs:
            fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
            for row in coeff_jobs:
                if row.get("coeffs_E") is not None:
                    coeffs = np.abs(np.asarray(row["coeffs_E"]).ravel())
                    axs[0].semilogy(
                        np.arange(len(coeffs)),
                        np.maximum(coeffs, 1e-18),
                        "o-", ms=3, lw=1.0,
                        label=f"p={row['order']}, Q={best_q}",
                    )
                if row.get("coeffs_nu") is not None:
                    coeffs = np.abs(np.asarray(row["coeffs_nu"]).ravel())
                    axs[1].semilogy(
                        np.arange(len(coeffs)),
                        np.maximum(coeffs, 1e-18),
                        "o-", ms=3, lw=1.0,
                        label=f"p={row['order']}, Q={best_q}",
                    )
            axs[0].set_title("PCE coefficient sequence for E*")
            axs[0].set_xlabel("coefficient index i")
            axs[0].set_ylabel("|u_i|")
            axs[1].set_title("PCE coefficient sequence for ν*")
            axs[1].set_xlabel("coefficient index i")
            axs[1].set_ylabel("|u_i|")
            for ax in axs:
                ax.grid(alpha=0.3, which="both")
                ax.legend(fontsize=8)
            fig.suptitle(
                f"PCE coefficient sequence at {nx}×{nx} RVE",
                y=1.02,
            )
            fig.tight_layout()
            fname = f"scalar_pce_coefficients_RVE_{nx}x{nx}.png"
            fig.savefig(os.path.join(outdir, fname), dpi=150,
                        bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {fname}")


        coeff_p = 2 if 2 in orders else min(orders)
        coeff_q_jobs = sorted([r for r in nx_jobs if r["order"] == coeff_p],
                              key=q_of)
        if coeff_q_jobs:
            n_coeff_show = min(
                3,
                min(len(np.asarray(r["coeffs_E"]).ravel())
                    for r in coeff_q_jobs if r.get("coeffs_E") is not None),
            )
            if n_coeff_show > 0:
                fig, axs = plt.subplots(2, n_coeff_show,
                                        figsize=(4.2 * n_coeff_show, 6.8),
                                        sharex=True)
                if n_coeff_show == 1:
                    axs = np.asarray(axs).reshape(2, 1)
                xs = [q_of(r) for r in coeff_q_jobs]
                for i in range(n_coeff_show):
                    ys = [np.asarray(r["coeffs_E"]).ravel()[i]
                          for r in coeff_q_jobs]
                    axs[0, i].plot(xs, ys, "o-", lw=1.5)
                    axs[0, i].set_title(f"E*: coefficient u{i}")
                    axs[0, i].set_ylabel(f"u{i}")

                    ys = [np.asarray(r["coeffs_nu"]).ravel()[i]
                          for r in coeff_q_jobs]
                    axs[1, i].plot(xs, ys, "o-", lw=1.5)
                    axs[1, i].set_title(f"ν*: coefficient u{i}")
                    axs[1, i].set_ylabel(f"u{i}")
                    axs[1, i].set_xlabel("quadrature depth Q")

                for ax in axs.flat:
                    ax.set_xticks(quad_orders)
                    ax.grid(alpha=0.3)
                fig.suptitle(
                    f"Coefficient estimator convergence, p={coeff_p}, {nx}×{nx} RVE",
                    y=1.02,
                )
                fig.tight_layout()
                fname = f"scalar_pce_coefficient_convergence_RVE_{nx}x{nx}.png"
                fig.savefig(os.path.join(outdir, fname), dpi=150,
                            bbox_inches="tight")
                plt.close(fig)
                print(f"  saved {fname}")


        fig, axs = plt.subplots(2, 2, figsize=(12, 8.2), sharex=True)
        for ax, (key, ylabel, title) in zip(axs.flat, moment_specs):
            for p in orders:
                series = sorted([r for r in nx_jobs if r["order"] == p],
                                key=q_of)
                xs = [q_of(r) for r in series]
                ys = [moment(r, key) for r in series]
                ax.plot(xs, ys, "o-", lw=1.6, ms=5, label=f"p={p}")
            if nx_mc is not None:
                ax.axhline(moment(nx_mc, key), color="k", ls="--", lw=1.1,
                           label=f"MC N={nx_mc['N']:,}")
            ax.set_title(title)
            ax.set_xlabel("quadrature depth Q")
            ax.set_ylabel(ylabel)
            ax.set_xticks(quad_orders)
            ax.grid(alpha=0.3)
        axs[0, 0].legend(fontsize=8, ncol=2)
        fig.suptitle(
            f"Inner convergence: quadrature depth Q at {nx}×{nx} RVE",
            y=1.01,
        )
        fig.tight_layout()
        fname = f"scalar_inner_convergence_RVE_{nx}x{nx}.png"
        fig.savefig(os.path.join(outdir, fname), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {fname}")


        fig, axs = plt.subplots(2, 2, figsize=(12, 8.2), sharex=True)
        for ax, (key, ylabel, title) in zip(axs.flat, moment_specs):
            for q in quad_orders:
                series = sorted([r for r in nx_jobs if q_of(r) == q],
                                key=lambda row: row["order"])
                xs = [r["order"] for r in series]
                ys = [moment(r, key) for r in series]
                ax.plot(xs, ys, "o-", lw=1.6, ms=5, label=f"Q={q}")
            if nx_mc is not None:
                ax.axhline(moment(nx_mc, key), color="k", ls="--", lw=1.1,
                           label=f"MC N={nx_mc['N']:,}")
            ax.set_title(title)
            ax.set_xlabel("PCE polynomial order p")
            ax.set_ylabel(ylabel)
            ax.set_xticks(orders)
            ax.grid(alpha=0.3)
        axs[0, 0].legend(fontsize=8, ncol=2)
        fig.suptitle(
            f"Outer convergence: PCE order p at {nx}×{nx} RVE",
            y=1.01,
        )
        fig.tight_layout()
        fname = f"scalar_outer_convergence_RVE_{nx}x{nx}.png"
        fig.savefig(os.path.join(outdir, fname), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {fname}")


def main():
    parser = argparse.ArgumentParser(
        description="Scalar (4-D) PCE inner/outer convergence sweep.")
    parser.add_argument("--out", type=str, default="scalar_results.pkl",
                        help="output pickle filename")
    parser.add_argument("--outdir", type=str, default="scalar_plots",
                        help="folder for plots")
    parser.add_argument("--N", type=int, default=100000,
                        help="MC sample count (default 100000)")
    parser.add_argument("--nx", type=int, default=None,
                        help="legacy shortcut: run one n×n RVE")
    parser.add_argument("--rves", type=int, nargs="+", default=[3, 5],
                        help="RVE sizes n for n×n RVEs (default 3 5)")
    parser.add_argument("--orders", type=int, nargs="+",
                        default=[2, 3, 4, 5, 6],
                        help="PCE orders p to run (default 2 3 4 5 6)")
    parser.add_argument("--quad-orders", type=int, nargs="+",
                        default=[2, 4, 6, 8],
                        help="quadrature depths Q to run (default 2 4 6 8)")
    parser.add_argument("--full-quadrature", action="store_true",
                        help="use full tensor quadrature instead of sparse Smolyak")
    parser.add_argument("--no-mc", action="store_true",
                        help="skip Monte Carlo reference runs")
    parser.add_argument("--pdf-samples", type=int, default=100000,
                        help="cheap PCE surrogate samples for density plots (default 100000)")
    args = parser.parse_args()
    rves = [args.nx] if args.nx is not None else args.rves
    sparse = not args.full_quadrature

    print("=" * 70)
    print("SCALAR (4-D) UQ — PCE inner/outer convergence sweep")
    print("=" * 70)
    print(f"RVE sizes        : {', '.join(f'{n}x{n}' for n in rves)}")
    print(f"MC samples       : {'skipped' if args.no_mc else args.N}")
    print(f"PCE orders p     : {args.orders}")
    print(f"Quadrature Q     : {args.quad_orders}")
    print(f"Quadrature grid  : {'sparse Smolyak' if sparse else 'full tensor'}")
    print(f"Output pickle    : {args.out}")
    print(f"Output plots     : {args.outdir}/")
    print("=" * 70, flush=True)

    spec = InputSpec()
    results = []
    quad_cache = {}
    total_t0 = time.time()
    total_jobs = len(rves) * (len(args.orders) * len(args.quad_orders)
                              + (0 if args.no_mc else 1))
    completed_jobs = 0

    for nx in rves:
        if not args.no_mc:
            try:
                results.append(run_mc(nx, args.N, spec))
            except Exception as e:
                print(f"!!! MC {nx}x{nx} CRASHED: {e}", flush=True)
                results.append({"kind": "MC_crash", "nx": nx,
                                "N": args.N, "error": str(e)})
            completed_jobs += 1
            with open(args.out, "wb") as f:
                pickle.dump(results, f)
            print(f"\n[saved {completed_jobs}/{total_jobs} jobs to {args.out}]",
                  flush=True)
            print(f"[cumulative wall time: {(time.time()-total_t0)/60:.2f} min]",
                  flush=True)

        for p in args.orders:
            for q in args.quad_orders:
                try:
                    results.append(run_pce(nx, p, q, spec, sparse=sparse,
                                           quad_cache=quad_cache,
                                           pdf_samples=args.pdf_samples))
                except Exception as e:
                    print(f"!!! PCE {nx}x{nx} p={p}, Q={q} CRASHED: {e}",
                          flush=True)
                    results.append({"kind": "PCE_crash", "nx": nx,
                                    "order": p, "quadrature_order": q,
                                    "error": str(e)})
                completed_jobs += 1
                with open(args.out, "wb") as f:
                    pickle.dump(results, f)
                print(f"\n[saved {completed_jobs}/{total_jobs} jobs to {args.out}]",
                      flush=True)
                print(f"[cumulative wall time: {(time.time()-total_t0)/60:.2f} min]",
                      flush=True)


    total_elapsed = time.time() - total_t0
    print("\n" + "=" * 70)
    print(f"SCALAR SWEEP DONE  --  total wall time: {total_elapsed/60:.2f} min")
    print("=" * 70)
    print()
    print(f"  {'kind':>12} {'RVE':>5} {'p':>3} {'Q':>3} {'Nq':>7} "
          f"{'mean E*':>14} {'Var E*':>14} {'mean nu*':>10} {'time (s)':>10}")
    print("-" * 94)
    for r in results:
        k = r.get("kind", "?")
        if k.endswith("_crash") or k == "PCE_failed":
            err_msg = (r.get("error") or "?")[:40]
            print(f"  {k:>10} {r.get('nx','?'):>2}x{r.get('nx','?'):<2} "
                  f"   --      (error: {err_msg})")
            continue
        tag = "PCE" if r["kind"] == "PCE" else f"MC N={r['N']}"
        p = r.get("order", "--")
        q = r.get("quadrature_order", "--")
        nq = r.get("n_quad", "--")
        print(f"  {tag:>12} {r['nx']:>2}x{r['nx']:<2} {str(p):>3} {str(q):>3} {str(nq):>7} "
              f"{r['mean_E']:>14.6e} {r['var_E']:>14.6e} "
              f"{r['mean_nu']:>10.4f} {r['elapsed_sec']:>10.1f}")

    print(f"\nResults saved to {args.out}")
    print(f"\nGenerating plots into {args.outdir}/ ...")
    make_plots(results, args.outdir, spec=spec)
    print("Done.")


if __name__ == "__main__":
    main()

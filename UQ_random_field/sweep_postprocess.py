from __future__ import annotations
import os
import tempfile
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(),
                                                  "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sstats
import pickle
import argparse
import warnings
import csv
from functools import lru_cache
warnings.filterwarnings("ignore")

from monte_carlo import InputSpec
from pce_field import PCEFieldResult, HermiteSurrogate
from hermite_fast import eval_surrogate_fast
from forward_map import build_field_problem

def load_results(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def filter_jobs(results, kind, nx=None, order=None):
    out = []
    for r in results:
        if r.get("kind") != kind:
            continue
        if nx is not None and r.get("nx") != nx:
            continue
        if order is not None and r.get("order") != order:
            continue
        out.append(r)
    return out


def finite_values(job, key):
    values = job.get(key)
    if values is None:
        return np.array([], dtype=float)
    values = np.asarray(values, dtype=float).ravel()
    return values[np.isfinite(values)]


def n_samples(job):
    if "N" in job:
        return int(job["N"])
    values = finite_values(job, "E_star")
    return len(values)


def raw_second_moment(job, field="E"):
    sample_key = "E_star" if field == "E" else "nu_star"
    mean_key = f"mean_{field}"
    var_key = f"var_{field}"

    values = finite_values(job, sample_key)
    if len(values):
        return float(np.mean(values ** 2))

    mean = job.get(mean_key)
    var = job.get(var_key)
    if mean is None or var is None:
        return np.nan
    return float(mean ** 2 + var)


def sample_variance(job, field="E"):
    sample_key = "E_star" if field == "E" else "nu_star"
    values = finite_values(job, sample_key)
    if len(values) > 1:
        return float(np.var(values, ddof=1))
    return float(job.get(f"var_{field}", np.nan))


def sample_mean(job, field="E"):
    sample_key = "E_star" if field == "E" else "nu_star"
    values = finite_values(job, sample_key)
    if len(values):
        return float(np.mean(values))
    return float(job.get(f"mean_{field}", np.nan))


def running_moments(values, max_points=250):
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    idx = np.unique(np.rint(np.geomspace(1, n, min(max_points, n))).astype(int))
    csum = np.cumsum(values)
    csum2 = np.cumsum(values ** 2)
    mean = csum[idx - 1] / idx
    second = csum2[idx - 1] / idx
    return idx, mean, second


def plot_kde_or_hist(ax, values, xgrid=None, **kwargs):
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None

    if xgrid is None:
        lo, hi = np.percentile(values, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = values.min(), values.max()
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        xgrid = np.linspace(lo - pad, hi + pad, 400)

    try:
        kde = sstats.gaussian_kde(values)
        line, = ax.plot(xgrid, kde(xgrid), **kwargs)
        return line
    except Exception:
        ax.hist(values, bins="auto", density=True, histtype="step", **kwargs)
        return ax


def pce_quadrature_depth(job):
    return int(job.get("n_quad", job.get("n_basis", 0)))


def pce_quad_order(job):
    return int(job.get("quad_order", job.get("order", 0)))


def positive_floor(values, floor=1e-16):
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values) & (values > floor), values, floor)


@lru_cache(maxsize=None)
def _basis(d, p):
    from hermite_fast import gen_multiindices, compute_norms
    mi = gen_multiindices(d, p)
    return mi, compute_norms(mi)


def sample_pce_surrogate(pce_job, N=30000, seed=2026, xi=None):
    d = pce_job["n_dim"]
    p = pce_job["order"]
    coeffs_E = pce_job["coeffs_E"]
    coeffs_nu = pce_job["coeffs_nu"]
    mi, norms = _basis(d, p)

    if xi is None:
        rng = np.random.default_rng(seed)
        xi = rng.standard_normal((d, N))
    E  = eval_surrogate_fast(xi, coeffs_E, mi, norms, p)
    nu = eval_surrogate_fast(xi, coeffs_nu, mi, norms, p)
    return np.asarray(E).ravel(), np.asarray(nu).ravel()


def kl_cumulative(field):
    eigvals = np.asarray(field.eigvals, dtype=float)
    idx = np.arange(1, len(eigvals) + 1)
    cumulative = np.cumsum(eigvals) / float(field.total_var)
    return idx, cumulative


def plot_kl_convergence(rve_sizes, outdir, ell_c=1.5,
                        sigma_delta_ratio=0.03, energy_tol=0.95):
    if not rve_sizes:
        return

    fig, axs = plt.subplots(2, 2, figsize=(12.0, 8.0), squeeze=False)
    fields = [
        ("dx", "delta_x"),
        ("dy", "delta_y"),
        ("t", "g_t"),
        ("E", "g_E"),
    ]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(rve_sizes)))

    for color, nx in zip(cmap, rve_sizes):
        fp = build_field_problem(nx=nx, ny=nx,
                                 ell_c=ell_c,
                                 sigma_delta_ratio=sigma_delta_ratio,
                                 energy_tol=energy_tol)
        for ax, (attr, label) in zip(axs.ravel(), fields):
            field = getattr(fp.kl_model, attr)
            idx, cumulative = kl_cumulative(field)
            ax.plot(idx, 100.0 * cumulative, marker="o", ms=3.5, lw=1.3,
                    color=color, label=f"{nx}x{nx} (K={field.n_dim})")
            ax.scatter([field.n_dim], [field.energy_frac * 100.0],
                       s=25, color=color, zorder=3)
            ax.set_title(label)
            ax.set_xlabel("retained KL modes")
            ax.set_ylabel("captured variance [%]")
            ax.grid(alpha=0.3)

    for ax in axs.ravel():
        ax.axhline(energy_tol * 100.0, color="k", ls="--", lw=0.9,
                   alpha=0.55, label=f"{energy_tol*100:.0f}% target")
        ax.set_ylim(0, 102)
        handles, labels = ax.get_legend_handles_labels()
        dedup = dict(zip(labels, handles))
        ax.legend(dedup.values(), dedup.keys(), fontsize=7, loc="best")

    fig.suptitle("K-L truncation convergence", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "sweep_kl_convergence.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved sweep_kl_convergence.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", type=str, default="results_sweep.pkl")
    parser.add_argument("--outdir", type=str, default="plots_modified")
    parser.add_argument("--pdf-samples", type=int, default=30000,
                        help="cheap PCE surrogate samples used for PDF plots")
    parser.add_argument("--ell-c", type=float, default=1.5,
                        help="KL correlation length used for the KL plot")
    parser.add_argument("--sigma-delta", type=float, default=0.03,
                        help="node perturbation std / nominal length for KL plot")
    parser.add_argument("--energy-tol", type=float, default=0.95,
                        help="KL retained-energy target for the KL plot")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    results = load_results(args.inp)
    print(f"Loaded {len(results)} jobs from {args.inp}")

    mc_jobs  = sorted(filter_jobs(results, "MC"),  key=lambda r: r["nx"])
    pce_jobs_all = sorted(filter_jobs(results, "PCE"),
                          key=lambda r: (r["nx"], r["order"], pce_quad_order(r)))
    pce_jobs = pce_jobs_all

    mc_by_nx = {}
    for r in mc_jobs:
        mc_by_nx.setdefault(r["nx"], r)
    pce_by_nx = {}
    for r in pce_jobs:
        pce_by_nx.setdefault(r["nx"], []).append(r)

    print(f"  MC  jobs:  {[r['nx'] for r in mc_jobs]}")
    print(f"  PCE jobs:  {[(r['nx'], r['order'], pce_quad_order(r)) for r in pce_jobs_all]}")

    all_rve_sizes = sorted(set([r["nx"] for r in mc_jobs] +
                               [r["nx"] for r in pce_jobs_all]))

    if mc_jobs:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))
        cmap = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(mc_jobs), 1)))

        for color, r in zip(cmap, mc_jobs):
            values = finite_values(r, "E_star")
            if len(values) < 2:
                continue
            plot_kde_or_hist(ax, values, color=color, lw=1.8,
                             label=f"{r['nx']}x{r['nx']} (N={len(values):,})")

        ax.set_xlabel("E*")
        ax.set_ylabel("density")
        ax.set_title("Monte Carlo density of E*")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "sweep_mc_density.png"),
                    dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("  saved sweep_mc_density.png")

    if mc_jobs:
        fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.0))
        cmap = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(mc_jobs), 1)))

        for color, r in zip(cmap, mc_jobs):
            label = f"{r['nx']}x{r['nx']}"

            values_E = finite_values(r, "E_star")
            idx_E, run_mean_E, run_m2_E = running_moments(values_E)
            if len(idx_E):
                axs[0, 0].semilogx(idx_E, run_mean_E, color=color, lw=1.6,
                                    label=label)
                axs[0, 0].axhline(run_mean_E[-1], color=color, lw=0.9,
                                   ls="--", alpha=0.45)
                axs[0, 1].semilogx(idx_E, run_m2_E, color=color, lw=1.6,
                                    label=label)
                axs[0, 1].axhline(run_m2_E[-1], color=color, lw=0.9,
                                   ls="--", alpha=0.45)

            values_nu = finite_values(r, "nu_star")
            idx_nu, run_mean_nu, run_m2_nu = running_moments(values_nu)
            if len(idx_nu):
                axs[1, 0].semilogx(idx_nu, run_mean_nu, color=color, lw=1.6,
                                    label=label)
                axs[1, 0].axhline(run_mean_nu[-1], color=color, lw=0.9,
                                   ls="--", alpha=0.45)
                axs[1, 1].semilogx(idx_nu, run_m2_nu, color=color, lw=1.6,
                                    label=label)
                axs[1, 1].axhline(run_m2_nu[-1], color=color, lw=0.9,
                                   ls="--", alpha=0.45)

        axs[0, 0].set_ylabel("running E[E*]")
        axs[0, 0].set_title("E* mean estimator")
        axs[0, 1].set_ylabel("running E[(E*)^2]")
        axs[0, 1].set_title("E* second raw moment")
        axs[1, 0].set_ylabel("running E[nu*]")
        axs[1, 0].set_title("nu* mean estimator")
        axs[1, 1].set_ylabel("running E[(nu*)^2]")
        axs[1, 1].set_title("nu* second raw moment")
        for ax in axs.ravel():
            ax.set_xlabel("MC samples")
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "sweep_mc_moment_convergence.png"),
                    dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("  saved sweep_mc_moment_convergence.png")

    plot_kl_convergence(all_rve_sizes, args.outdir,
                        ell_c=args.ell_c,
                        sigma_delta_ratio=args.sigma_delta,
                        energy_tol=args.energy_tol)

    if len(mc_jobs) >= 2:
        nxs   = [r["nx"] for r in mc_jobs]
        means = [sample_mean(r, "E") for r in mc_jobs]
        vars_ = [sample_variance(r, "E") for r in mc_jobs]
        stds  = [np.sqrt(v) for v in vars_]
        covs  = [s / m * 100 for s, m in zip(stds, means)]
        SEs   = [s / np.sqrt(n_samples(r)) for s, r in zip(stds, mc_jobs)]

        fig, axs = plt.subplots(1, 3, figsize=(14, 4.0))

        axs[0].errorbar(nxs, means, yerr=[2*se for se in SEs],
                        marker="o", ms=7, lw=1.5, capsize=4)
        axs[0].set_xlabel("RVE size n  (for n×n RVE)")
        axs[0].set_ylabel("E[E*]")
        axs[0].set_title("Mean of E* vs RVE size")
        axs[0].set_xticks(nxs); axs[0].grid(alpha=0.3)

        axs[1].plot(nxs, vars_, marker="o", ms=7, lw=1.5)
        axs[1].set_xlabel("RVE size n")
        axs[1].set_ylabel("Var[E*]")
        axs[1].set_title("Variance of E* vs RVE size")
        axs[1].set_xticks(nxs); axs[1].grid(alpha=0.3)

        axs[2].plot(nxs, covs, marker="o", ms=7, lw=1.5, color="C1")
        axs[2].set_xlabel("RVE size n")
        axs[2].set_ylabel("CoV(E*)  [%]")
        axs[2].set_title("CoV of E* vs RVE size  (RVE convergence)")
        axs[2].set_xticks(nxs); axs[2].grid(alpha=0.3)

        fig.suptitle(f"Monte Carlo — RVE size convergence  (N = {n_samples(mc_jobs[0]):,})",
                     y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "sweep_rve_convergence.png"),
                    dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("  saved sweep_rve_convergence.png")

    rve_sizes = sorted(set(r["nx"] for r in pce_jobs))
    order_groups = {}
    for r in pce_jobs_all:
        order_groups.setdefault((r["nx"], pce_quad_order(r)), []).append(r)
    order_groups = {
        key: sorted(group, key=lambda rr: rr["order"])
        for key, group in order_groups.items()
        if len({rr["order"] for rr in group}) > 1
    }

    if order_groups:
        fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.5))
        cmap = plt.cm.viridis(np.linspace(0.12, 0.88, max(len(order_groups), 1)))

        for color, ((nx, q), group) in zip(cmap, sorted(order_groups.items())):
            orders  = [r["order"] for r in group]
            means_p = [sample_mean(r, "E") for r in group]
            m2_p    = [raw_second_moment(r, "E") for r in group]
            vars_p  = [sample_variance(r, "E") for r in group]
            label = f"{nx}x{nx}, Q={q}"

            axs[0].plot(orders, means_p, marker="o", ms=7, lw=1.5,
                        color=color, label=label)
            axs[1].plot(orders, m2_p, marker="o", ms=7, lw=1.5,
                        color=color, label=label)
            axs[2].plot(orders, vars_p, marker="o", ms=7, lw=1.5,
                        color=color, label=label)

            m = mc_by_nx.get(nx)
            if m is not None:
                axs[0].axhline(sample_mean(m, "E"), color=color, lw=1,
                               ls="--", alpha=0.5)
                axs[1].axhline(raw_second_moment(m, "E"), color=color, lw=1,
                               ls="--", alpha=0.5)
                axs[2].axhline(sample_variance(m, "E"), color=color, lw=1,
                               ls="--", alpha=0.5)

        axs[0].set_xlabel("PCE polynomial order p")
        axs[0].set_ylabel("E[E*]")
        axs[0].set_title("Outer-loop mean convergence")
        axs[0].grid(alpha=0.3); axs[0].legend(fontsize=7)

        axs[1].set_xlabel("PCE polynomial order p")
        axs[1].set_ylabel("E[(E*)^2]")
        axs[1].set_title("Outer-loop second moment convergence")
        axs[1].grid(alpha=0.3); axs[1].legend(fontsize=7)

        axs[2].set_xlabel("PCE polynomial order p")
        axs[2].set_ylabel("Var[E*]")
        axs[2].set_title("Outer-loop variance convergence")
        axs[2].grid(alpha=0.3); axs[2].legend(fontsize=7)

        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "sweep_pce_order_convergence.png"),
                    dpi=130, bbox_inches="tight")
        fig.savefig(os.path.join(args.outdir, "sweep_pce_convergence.png"),
                    dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("  saved sweep_pce_order_convergence.png")

    if pce_jobs:
        n_rve = len(rve_sizes)
        fig, axs = plt.subplots(1, n_rve, figsize=(4.2 * n_rve, 4.0),
                                squeeze=False)
        axs = axs[0]

        for ax, nx in zip(axs, rve_sizes):
            per_rve = pce_by_nx[nx]
            for r in per_rve:
                cE = np.abs(r["coeffs_E"])
                ax.semilogy(np.arange(len(cE)), np.maximum(cE, 1e-16),
                            ".", ms=1.5, label=f"p={r['order']}")
            ax.set_xlabel("basis index α")
            ax.set_ylabel("|c_α|")
            ax.set_title(f"RVE {nx}×{nx}  (d = {per_rve[0]['n_dim']})")
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=8)
        fig.suptitle("PCE coefficient magnitudes for E* across sweep", y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "sweep_coefficients.png"),
                    dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("  saved sweep_coefficients.png")

    if mc_jobs and pce_jobs:
        n_rve = len(mc_jobs)
        fig, axs = plt.subplots(1, n_rve, figsize=(4.5 * n_rve, 4.0),
                                squeeze=False)
        axs = axs[0]

        for ax, mc_job in zip(axs, mc_jobs):
            nx = mc_job["nx"]
            E_mc = finite_values(mc_job, "E_star")
            if len(E_mc) < 10:
                continue

            lo, hi = np.percentile(E_mc, [0.5, 99.5])
            pad = 0.05 * (hi - lo) if hi > lo else 1.0
            xgrid = np.linspace(lo - pad, hi + pad, 400)
            plot_kde_or_hist(ax, E_mc, xgrid=xgrid, color="r", lw=2,
                             label=f"MC (N={len(E_mc):,})")

            pces_here = pce_by_nx.get(nx, [])

            xi_shared = None
            if pces_here:
                d_here = pces_here[0]["n_dim"]
                xi_shared = np.random.default_rng(2026).standard_normal(
                    (d_here, args.pdf_samples))

            for r in pces_here:
                try:
                    E_samp, _ = sample_pce_surrogate(r, xi=xi_shared)
                    plot_kde_or_hist(ax, E_samp, xgrid=xgrid, lw=1.3,
                                     label=f"PCE p={r['order']}")
                except Exception as e:
                    print(f"    skipping PCE PDF for {nx}x{nx} p={r['order']}: {e}")

            ax.set_xlabel("E*")
            ax.set_ylabel("density")
            ax.set_title(f"RVE {nx}×{nx}")
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.suptitle("PDF of E*: MC vs PCE, per RVE size", y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "sweep_pdf_overlay.png"),
                    dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("  saved sweep_pdf_overlay.png")

    if mc_jobs and pce_jobs:
        n_rve = len(mc_jobs)
        fig, axs = plt.subplots(1, n_rve, figsize=(4.5 * n_rve, 4.0),
                                squeeze=False)
        axs = axs[0]

        for ax, mc_job in zip(axs, mc_jobs):
            nx = mc_job["nx"]
            nu_mc = finite_values(mc_job, "nu_star")
            if len(nu_mc) < 10:
                continue

            lo, hi = np.percentile(nu_mc, [0.5, 99.5])
            pad = 0.05 * (hi - lo) if hi > lo else 1.0
            xgrid = np.linspace(lo - pad, hi + pad, 400)
            plot_kde_or_hist(ax, nu_mc, xgrid=xgrid, color="r", lw=2,
                             label=f"MC (N={len(nu_mc):,})")

            pces_here = pce_by_nx.get(nx, [])
            xi_shared = None
            if pces_here:
                d_here = pces_here[0]["n_dim"]
                xi_shared = np.random.default_rng(2026).standard_normal(
                    (d_here, args.pdf_samples))

            for r in pces_here:
                try:
                    _, nu_samp = sample_pce_surrogate(r, xi=xi_shared)
                    plot_kde_or_hist(ax, nu_samp, xgrid=xgrid, lw=1.3,
                                     label=f"PCE p={r['order']}")
                except Exception as e:
                    print(f"    skipping PCE nu PDF for {nx}x{nx} p={r['order']}: {e}")

            ax.set_xlabel("nu*")
            ax.set_ylabel("density")
            ax.set_title(f"RVE {nx}×{nx}")
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.suptitle("PDF of nu*: MC vs PCE, per RVE size", y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "sweep_pdf_overlay_nu.png"),
                    dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("  saved sweep_pdf_overlay_nu.png")

    quad_groups = {}
    for r in pce_jobs_all:
        quad_groups.setdefault((r["nx"], r["order"]), []).append(r)
    quad_groups = {
        key: sorted(group, key=pce_quad_order)
        for key, group in quad_groups.items()
        if len({pce_quad_order(r) for r in group}) > 1
    }

    if quad_groups:
        groups = list(sorted(quad_groups.items()))
        n_groups = len(groups)
        ncols = min(2, n_groups)
        nrows = int(np.ceil(n_groups / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.3 * nrows),
                                squeeze=False)
        axs_flat = axs.ravel()

        for ax, ((nx, order), group) in zip(axs_flat, groups):
            mc_values = finite_values(mc_by_nx.get(nx, {}), "E_star")
            pce_samples = []

            d_here = group[0]["n_dim"]
            rng = np.random.default_rng(2026 + 1000 * nx + order)
            xi_shared = rng.standard_normal((d_here, args.pdf_samples))
            for r in group:
                try:
                    E_samp, _ = sample_pce_surrogate(r, xi=xi_shared)
                    pce_samples.append((pce_quad_order(r), E_samp))
                except Exception as e:
                    print(f"    skipping quadrature PDF for {nx}x{nx} "
                          f"p={order}, Q={pce_quad_order(r)}: {e}")

            x_pool = [v for _, v in pce_samples if len(v)]
            if len(mc_values):
                x_pool.append(mc_values)
            if not x_pool:
                continue
            x_all = np.concatenate(x_pool)
            x_all = x_all[np.isfinite(x_all)]
            lo, hi = np.percentile(x_all, [0.5, 99.5])
            pad = 0.06 * (hi - lo) if hi > lo else 1.0
            xgrid = np.linspace(lo - pad, hi + pad, 500)

            if len(mc_values) >= 2:
                plot_kde_or_hist(ax, mc_values, xgrid=xgrid, color="red",
                                 lw=1.4, label="Monte Carlo")

            colors = plt.cm.plasma(np.linspace(0.15, 0.85, max(len(pce_samples), 1)))
            for color, (q, values) in zip(colors, pce_samples):
                plot_kde_or_hist(ax, values, xgrid=xgrid, color=color,
                                 lw=1.1, label=f"Q = {q}")

            ax.set_xlabel("E*")
            ax.set_ylabel("density")
            ax.set_title(f"Inner-loop quadrature convergence, "
                         f"{nx}x{nx}, p={order}")
            ax.grid(alpha=0.28)
            ax.legend(fontsize=8)

        for ax in axs_flat[len(groups):]:
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "sweep_pce_quad_pdf.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  saved sweep_pce_quad_pdf.png")

    order_pdf_groups = {}
    for r in pce_jobs_all:
        order_pdf_groups.setdefault((r["nx"], pce_quad_order(r)), []).append(r)
    order_pdf_groups = {
        key: sorted(group, key=lambda rr: rr["order"])
        for key, group in order_pdf_groups.items()
        if len({rr["order"] for rr in group}) > 1
    }

    if order_pdf_groups:
        groups = list(sorted(order_pdf_groups.items()))
        n_groups = len(groups)
        ncols = min(2, n_groups)
        nrows = int(np.ceil(n_groups / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.3 * nrows),
                                squeeze=False)
        axs_flat = axs.ravel()

        for ax, ((nx, q), group) in zip(axs_flat, groups):
            mc_values = finite_values(mc_by_nx.get(nx, {}), "E_star")
            pce_samples = []

            d_here = group[0]["n_dim"]
            rng = np.random.default_rng(4040 + 1000 * nx + q)
            xi_shared = rng.standard_normal((d_here, args.pdf_samples))
            for r in group:
                try:
                    E_samp, _ = sample_pce_surrogate(r, xi=xi_shared)
                    pce_samples.append((r["order"], E_samp))
                except Exception as e:
                    print(f"    skipping order PDF for {nx}x{nx} "
                          f"p={r['order']}, Q={q}: {e}")

            x_pool = [v for _, v in pce_samples if len(v)]
            if len(mc_values):
                x_pool.append(mc_values)
            if not x_pool:
                continue
            x_all = np.concatenate(x_pool)
            x_all = x_all[np.isfinite(x_all)]
            lo, hi = np.percentile(x_all, [0.5, 99.5])
            pad = 0.06 * (hi - lo) if hi > lo else 1.0
            xgrid = np.linspace(lo - pad, hi + pad, 500)

            if len(mc_values) >= 2:
                plot_kde_or_hist(ax, mc_values, xgrid=xgrid, color="red",
                                 lw=1.4, label="Monte Carlo")

            colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(pce_samples), 1)))
            for color, (order, values) in zip(colors, pce_samples):
                plot_kde_or_hist(ax, values, xgrid=xgrid, color=color,
                                 lw=1.1, label=f"p = {order}")

            ax.set_xlabel("E*")
            ax.set_ylabel("density")
            ax.set_title(f"Outer-loop order convergence, "
                         f"{nx}x{nx}, Q={q}")
            ax.grid(alpha=0.28)
            ax.legend(fontsize=8)

        for ax in axs_flat[len(groups):]:
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "sweep_pce_order_pdf.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  saved sweep_pce_order_pdf.png")

    if mc_jobs and pce_jobs:
        fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
        cmap = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(rve_sizes), 1)))

        for color, nx in zip(cmap, rve_sizes):
            m = mc_by_nx.get(nx)
            pce_match = pce_by_nx.get(nx, [])
            if m is None or not pce_match:
                continue

            ref_mean = sample_mean(m, "E")
            ref_m2 = raw_second_moment(m, "E")

            costs = [pce_quadrature_depth(r) for r in pce_match]
            err_mean = [
                abs(sample_mean(r, "E") - ref_mean) / abs(ref_mean)
                if ref_mean else np.nan
                for r in pce_match
            ]
            err_m2 = [
                abs(raw_second_moment(r, "E") - ref_m2) / abs(ref_m2)
                if ref_m2 else np.nan
                for r in pce_match
            ]
            labels = [f"p={r['order']}" for r in pce_match]

            plot_costs = positive_floor(costs, floor=1.0)
            plot_err_mean = positive_floor(err_mean)
            plot_err_m2 = positive_floor(err_m2)

            axs[0].loglog(plot_costs, plot_err_mean, "-o",
                          ms=7, lw=1.5, color=color, label=f"{nx}x{nx}")
            axs[1].loglog(plot_costs, plot_err_m2, "-o",
                          ms=7, lw=1.5, color=color, label=f"{nx}x{nx}")

            for ax, errs in zip(axs, [plot_err_mean, plot_err_m2]):
                for cost, err, label in zip(plot_costs, errs, labels):
                    ax.annotate(label, (cost, err),
                                textcoords="offset points", xytext=(4, 4),
                                fontsize=7, color=color)

        axs[0].set_ylabel("relative error in E[E*] vs MC")
        axs[0].set_title("Mean error vs PCE model calls")
        axs[1].set_ylabel("relative error in E[(E*)^2] vs MC")
        axs[1].set_title("Second-moment error vs PCE model calls")
        for ax in axs:
            ax.set_xlabel("model calls (PCE quadrature points)")
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=8, loc="best")

        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "sweep_efficiency.png"),
                    dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("  saved sweep_efficiency.png")

    txt_path = os.path.join(args.outdir, "sweep_summary.txt")
    csv_path = os.path.join(args.outdir, "sweep_summary.csv")
    summary_rows = []

    for r in results:
        k = r.get("kind", "?")
        if k.endswith("_crash") or k == "PCE_failed":
            summary_rows.append({
                "kind": k,
                "rve": f"{r.get('nx', '?')}x{r.get('ny', r.get('nx', '?'))}",
                "nx": r.get("nx", ""),
                "ny": r.get("ny", r.get("nx", "")),
                "n_dim": r.get("n_dim", ""),
                "param": r.get("order", r.get("N", r.get("arg", ""))),
                "quad_order": r.get("quad_order", ""),
                "model_calls": "",
                "mean_E": "",
                "second_moment_E": "",
                "var_E": "",
                "cov_E_percent": "",
                "mean_nu": "",
                "second_moment_nu": "",
                "var_nu": "",
                "elapsed_sec": r.get("elapsed_sec", ""),
                "rel_mean_E_error_vs_MC": "",
                "rel_second_moment_E_error_vs_MC": "",
                "status": r.get("error", "?"),
            })
            continue

        mean_E = sample_mean(r, "E")
        var_E = sample_variance(r, "E")
        m2_E = raw_second_moment(r, "E")
        mean_nu = sample_mean(r, "nu")
        var_nu = sample_variance(r, "nu")
        m2_nu = raw_second_moment(r, "nu")
        cov_E = np.sqrt(var_E) / abs(mean_E) * 100 if mean_E else np.nan
        model_calls = pce_quadrature_depth(r) if k == "PCE" else n_samples(r)

        rel_mean = ""
        rel_m2 = ""
        mc_ref = mc_by_nx.get(r.get("nx"))
        if k == "PCE" and mc_ref is not None:
            ref_mean = sample_mean(mc_ref, "E")
            ref_m2 = raw_second_moment(mc_ref, "E")
            rel_mean = abs(mean_E - ref_mean) / abs(ref_mean) if ref_mean else np.nan
            rel_m2 = abs(m2_E - ref_m2) / abs(ref_m2) if ref_m2 else np.nan

        summary_rows.append({
            "kind": k,
            "rve": f"{r['nx']}x{r.get('ny', r['nx'])}",
            "nx": r["nx"],
            "ny": r.get("ny", r["nx"]),
            "n_dim": r.get("n_dim", ""),
            "param": r.get("order", r.get("N", "")),
            "quad_order": pce_quad_order(r) if k == "PCE" else "",
            "model_calls": model_calls,
            "mean_E": mean_E,
            "second_moment_E": m2_E,
            "var_E": var_E,
            "cov_E_percent": cov_E,
            "mean_nu": mean_nu,
            "second_moment_nu": m2_nu,
            "var_nu": var_nu,
            "elapsed_sec": r.get("elapsed_sec", ""),
            "rel_mean_E_error_vs_MC": rel_mean,
            "rel_second_moment_E_error_vs_MC": rel_m2,
            "status": "ok",
        })

    with open(txt_path, "w") as f:
        print("\n" + "=" * 125)
        print("SUMMARY — RVE size × method sweep")
        print("=" * 125)
        f.write("SUMMARY — RVE size × method sweep\n")
        f.write("=" * 125 + "\n")

        header = (f"  {'kind':>16} {'RVE':>5} {'dim':>5} {'param':>7} "
                  f"{'Q':>4} "
                  f"{'calls':>9} {'mean E*':>14} {'E[(E*)^2]':>14} "
                  f"{'Var E*':>14} {'CoV %':>8} {'mean nu*':>10} "
                  f"{'err mean':>10} {'err m2':>10} {'time (s)':>10}")
        print(header); f.write(header + "\n")
        print("-" * 125); f.write("-" * 125 + "\n")

        for row in summary_rows:
            if row["status"] != "ok":
                msg = (f"  {row['kind']:>16} {row['rve']:>5} "
                       f"   --       --      (error: {str(row['status'])[:45]})")
                print(msg); f.write(msg + "\n")
                continue
            tag = (f"PCE p={row['param']},Q={row['quad_order']}"
                   if row["kind"] == "PCE" else f"MC N={row['param']}")
            q_s = str(row["quad_order"]) if row["quad_order"] != "" else "--"
            err_mean = row["rel_mean_E_error_vs_MC"]
            err_m2 = row["rel_second_moment_E_error_vs_MC"]
            err_mean_s = f"{err_mean:10.3e}" if err_mean != "" else f"{'--':>10}"
            err_m2_s = f"{err_m2:10.3e}" if err_m2 != "" else f"{'--':>10}"
            line = (f"  {tag:>16} {row['rve']:>5} "
                    f"{int(row['n_dim']):>5d} {str(row['param']):>7} "
                    f"{q_s:>4} "
                    f"{int(row['model_calls']):>9d} "
                    f"{row['mean_E']:>14.6e} {row['second_moment_E']:>14.6e} "
                    f"{row['var_E']:>14.6e} {row['cov_E_percent']:>8.3f} "
                    f"{row['mean_nu']:>10.4f} {err_mean_s} {err_m2_s} "
                    f"{float(row['elapsed_sec']):>10.1f}")
            print(line); f.write(line + "\n")

    if summary_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        with open(csv_path, "w", newline="") as f:
            f.write("status\nno jobs found\n")

    print(f"\n  saved sweep_summary.txt")
    print(f"  saved sweep_summary.csv")


if __name__ == "__main__":
    main()

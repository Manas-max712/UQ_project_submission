from __future__ import annotations

import argparse
import os
import pickle
import tempfile
import time
import warnings

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(),
                                                  "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sstats

from forward_map import build_field_problem
from mc_field import run_mc_field
from pce_field import build_pce_field
from hermite_fast import gen_multiindices, compute_norms, eval_surrogate_fast

warnings.filterwarnings("ignore")


def sparse_point_count(d: int, q: int) -> int:
    dp = [0] * (q + 1)
    dp[0] = 1
    for _ in range(d):
        ndp = [0] * (q + 1)
        for total, value in enumerate(dp):
            if value == 0:
                continue
            for level in range(q - total + 1):
                ndp[total + level] += value * (level + 1)
        dp = ndp
    return int(sum(dp))


def load_state(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {"mc": None, "pce": {}, "config": {}}


def save_state(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp, path)


def compact_pce_result(result) -> dict:
    return {
        "kind": "PCE",
        "nx": 2,
        "ny": 2,
        "order": result.order,
        "quad_order": result.quad_order,
        "n_dim": result.n_dim,
        "n_basis": result.n_basis,
        "n_quad": result.n_quad,
        "coeffs_E": result.coeffs_E,
        "coeffs_nu": result.coeffs_nu,
        "mean_E": result.mean_E,
        "var_E": result.var_E,
        "mean_nu": result.mean_nu,
        "var_nu": result.var_nu,
        "elapsed_sec": result.elapsed_sec,
    }


def finite(values) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    return values[np.isfinite(values)]


def plot_kde(ax, values, xgrid, **kwargs) -> None:
    values = finite(values)
    if len(values) < 2:
        return
    try:
        kde = sstats.gaussian_kde(values)
        ax.plot(xgrid, kde(xgrid), **kwargs)
    except Exception:
        ax.hist(values, bins="auto", density=True, histtype="step", **kwargs)


def sample_pce(job: dict, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = int(job["order"])
    d = int(job["n_dim"])
    if xi.shape[0] != d:
        raise ValueError(f"xi has d={xi.shape[0]}, expected {d}")
    multi_indices = gen_multiindices(d, p)
    norms = compute_norms(multi_indices)
    E = eval_surrogate_fast(xi, job["coeffs_E"], multi_indices, norms, p)
    nu = eval_surrogate_fast(xi, job["coeffs_nu"], multi_indices, norms, p)
    return np.asarray(E).ravel(), np.asarray(nu).ravel()


def run_study(args) -> dict:
    state = load_state(args.out)
    state["config"] = {
        "nx": 2,
        "ny": 2,
        "order": args.order,
        "q_orders": list(args.q_orders),
        "N_mc": args.N_mc,
        "seed": args.seed,
    }

    fp = build_field_problem(nx=2, ny=2,
                             ell_c=args.ell_c,
                             sigma_delta_ratio=args.sigma_delta,
                             energy_tol=args.energy_tol)

    print("=" * 72)
    print("INNER-LOOP CONVERGENCE: 2x2 RVE")
    print("=" * 72)
    print(f"Fixed PCE order p = {args.order}")
    print(f"Quadrature levels Q = {list(args.q_orders)}")
    print(f"Stochastic dimension d = {fp.n_dim}")
    print()
    for q in args.q_orders:
        print(f"  Q={q}: approx {sparse_point_count(fp.n_dim, q):,} quadrature points")
    print("=" * 72, flush=True)

    if state.get("mc") is None or args.force:
        print("\n# Running MC reference", flush=True)
        t0 = time.time()
        mc = run_mc_field(N=args.N_mc, fp=fp, seed=args.seed, verbose=True)
        state["mc"] = {
            "kind": "MC",
            "nx": 2,
            "ny": 2,
            "N": args.N_mc,
            "n_dim": mc.n_dim,
            "E_star": mc.E_star,
            "nu_star": mc.nu_star,
            "mean_E": float(np.nanmean(mc.E_star)),
            "var_E": float(np.nanvar(mc.E_star, ddof=1)),
            "mean_nu": float(np.nanmean(mc.nu_star)),
            "var_nu": float(np.nanvar(mc.nu_star, ddof=1)),
            "elapsed_sec": time.time() - t0,
        }
        save_state(args.out, state)
        print(f"[saved MC to {args.out}]", flush=True)
    else:
        print("\n# MC reference already present; use --force to recompute")

    for q in args.q_orders:
        key = str(q)
        if key in state["pce"] and not args.force:
            print(f"\n# PCE p={args.order}, Q={q} already present; skipping")
            continue

        print(f"\n# Running PCE p={args.order}, Q={q}", flush=True)
        result = build_pce_field(args.order, fp, quad_order=q, verbose=True)
        state["pce"][key] = compact_pce_result(result)
        save_state(args.out, state)
        print(f"[saved PCE p={args.order}, Q={q} to {args.out}]", flush=True)
        del result

    return state


def plot_results(state: dict, outdir: str, pdf_samples: int, seed: int) -> None:
    os.makedirs(outdir, exist_ok=True)

    mc = state.get("mc")
    pce_jobs = [state["pce"][q] for q in sorted(state.get("pce", {}),
                                                key=lambda x: int(x))]
    if mc is None or not pce_jobs:
        print("Nothing to plot yet: need MC and at least one PCE result.")
        return

    E_mc = finite(mc["E_star"])
    nu_mc = finite(mc["nu_star"])
    d = int(pce_jobs[0]["n_dim"])
    rng = np.random.default_rng(seed)
    xi_shared = rng.standard_normal((d, pdf_samples))

    E_samples = []
    nu_samples = []
    for job in pce_jobs:
        E_s, nu_s = sample_pce(job, xi_shared)
        E_samples.append((job["quad_order"], E_s))
        nu_samples.append((job["quad_order"], nu_s))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    pool = [E_mc] + [v for _, v in E_samples]
    x_all = finite(np.concatenate(pool))
    lo, hi = np.percentile(x_all, [0.5, 99.5])
    pad = 0.06 * (hi - lo) if hi > lo else 1.0
    xgrid = np.linspace(lo - pad, hi + pad, 500)

    plot_kde(ax, E_mc, xgrid, color="red", lw=1.8,
             label=f"Monte Carlo (N={len(E_mc):,})")
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(E_samples)))
    for color, (q, values) in zip(colors, E_samples):
        plot_kde(ax, values, xgrid, color=color, lw=1.2, label=f"Q = {q}")
    ax.set_xlabel("E*")
    ax.set_ylabel("density")
    ax.set_title("Inner-loop convergence, 2x2 RVE, p=2")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "inner_2x2_p2_E_pdf.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    pool = [nu_mc] + [v for _, v in nu_samples]
    x_all = finite(np.concatenate(pool))
    lo, hi = np.percentile(x_all, [0.5, 99.5])
    pad = 0.06 * (hi - lo) if hi > lo else 1.0
    xgrid = np.linspace(lo - pad, hi + pad, 500)

    plot_kde(ax, nu_mc, xgrid, color="red", lw=1.8,
             label=f"Monte Carlo (N={len(nu_mc):,})")
    for color, (q, values) in zip(colors, nu_samples):
        plot_kde(ax, values, xgrid, color=color, lw=1.2, label=f"Q = {q}")
    ax.set_xlabel("nu*")
    ax.set_ylabel("density")
    ax.set_title("Inner-loop convergence, 2x2 RVE, p=2")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "inner_2x2_p2_nu_pdf.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    qs = np.array([job["quad_order"] for job in pce_jobs], dtype=int)
    mean_E = np.array([job["mean_E"] for job in pce_jobs], dtype=float)
    var_E = np.array([job["var_E"] for job in pce_jobs], dtype=float)
    m2_E = mean_E ** 2 + var_E

    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.2))
    axs[0].plot(qs, mean_E, "-o", lw=1.5)
    axs[0].axhline(np.mean(E_mc), color="red", ls="--", lw=1.1,
                   label="MC")
    axs[0].set_xlabel("quadrature level Q")
    axs[0].set_ylabel("E[E*]")
    axs[0].set_title("Mean")

    axs[1].plot(qs, m2_E, "-o", lw=1.5)
    axs[1].axhline(np.mean(E_mc ** 2), color="red", ls="--", lw=1.1,
                   label="MC")
    axs[1].set_xlabel("quadrature level Q")
    axs[1].set_ylabel("E[(E*)^2]")
    axs[1].set_title("Second raw moment")

    axs[2].plot(qs, var_E, "-o", lw=1.5)
    axs[2].axhline(np.var(E_mc, ddof=1), color="red", ls="--", lw=1.1,
                   label="MC")
    axs[2].set_xlabel("quadrature level Q")
    axs[2].set_ylabel("Var[E*]")
    axs[2].set_title("Variance")

    for ax in axs:
        ax.grid(alpha=0.3)
        ax.set_xticks(qs)
        ax.legend(fontsize=8)
    fig.suptitle("Inner-loop moment convergence, 2x2 RVE, p=2", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "inner_2x2_p2_moments.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    txt_path = os.path.join(outdir, "inner_2x2_p2_summary.txt")
    with open(txt_path, "w") as f:
        header = (
            f"{'Q':>4} {'Nquad':>12} {'mean_E':>14} {'var_E':>14} "
            f"{'mean_nu':>12} {'var_nu':>12} {'time_s':>10}"
        )
        print(header)
        f.write(header + "\n")
        for job in pce_jobs:
            line = (
                f"{job['quad_order']:>4d} {job['n_quad']:>12d} "
                f"{job['mean_E']:>14.6e} {job['var_E']:>14.6e} "
                f"{job['mean_nu']:>12.6f} {job['var_nu']:>12.4e} "
                f"{job['elapsed_sec']:>10.1f}"
            )
            print(line)
            f.write(line + "\n")

    print(f"\nSaved plots and summary to {outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone 2x2 inner-loop convergence study."
    )
    parser.add_argument("--out", type=str, default="inner_2x2_p2.pkl",
                        help="incremental result pickle")
    parser.add_argument("--outdir", type=str, default="inner_2x2_p2_plots",
                        help="directory for plots")
    parser.add_argument("--order", type=int, default=2,
                        help="fixed PCE order p; default 2")
    parser.add_argument("--q-orders", type=int, nargs="+",
                        default=[2, 3, 4, 5, 6],
                        help="quadrature levels Q; default 2 3 4 5 6")
    parser.add_argument("--N-mc", type=int, default=10000,
                        help="MC reference sample count")
    parser.add_argument("--pdf-samples", type=int, default=30000,
                        help="cheap surrogate samples used for PDF plots")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ell-c", type=float, default=1.5)
    parser.add_argument("--sigma-delta", type=float, default=0.03)
    parser.add_argument("--energy-tol", type=float, default=0.95)
    parser.add_argument("--force", action="store_true",
                        help="recompute jobs even if present in --out")
    parser.add_argument("--plot-only", action="store_true",
                        help="skip solves and only regenerate plots from --out")
    args = parser.parse_args()

    if args.plot_only:
        state = load_state(args.out)
    else:
        state = run_study(args)
    plot_results(state, args.outdir, args.pdf_samples, args.seed)


if __name__ == "__main__":
    main()

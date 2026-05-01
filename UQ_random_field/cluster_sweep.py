from __future__ import annotations
import numpy as np
import pickle
import time
import warnings
import argparse
import sys
warnings.filterwarnings("ignore")

from forward_map import build_field_problem
from mc_field import run_mc_field
from pce_field import build_pce_field

def run_mc_for_rve(nx, N, ell_c=1.5, sigma_delta_ratio=0.03,
                   energy_tol=0.95, seed=2026):
    print(f"\n{'#' * 70}")
    print(f"# MC  --  RVE {nx}x{nx}  --  N = {N}")
    print(f"{'#' * 70}", flush=True)
    t0 = time.time()
    fp = build_field_problem(nx=nx, ny=nx,
                             ell_c=ell_c,
                             sigma_delta_ratio=sigma_delta_ratio,
                             energy_tol=energy_tol)
    print(f"  Built field problem: n_dim = {fp.n_dim}", flush=True)
    res = run_mc_field(N=N, fp=fp, seed=seed, verbose=True)
    elapsed = time.time() - t0
    print(f"\n  MC {nx}x{nx} took {elapsed:.1f}s total (including setup)",
          flush=True)
    return {
        "kind": "MC", "nx": nx, "ny": nx, "N": N, "n_dim": fp.n_dim,
        "E_star": res.E_star, "nu_star": res.nu_star,
        "xi_samples": res.xi_samples,
        "elapsed_sec": elapsed,
        "mean_E":  float(np.nanmean(res.E_star)),
        "var_E":   float(np.nanvar(res.E_star, ddof=1)),
        "mean_nu": float(np.nanmean(res.nu_star)),
        "var_nu":  float(np.nanvar(res.nu_star, ddof=1)),
    }


def run_pce_for_rve(nx, order, quad_order=None, ell_c=1.5, sigma_delta_ratio=0.03,
                    energy_tol=0.95):
    q = order if quad_order is None else quad_order
    print(f"\n{'#' * 70}")
    print(f"# PCE --  RVE {nx}x{nx}  --  order = {order}, Q = {q}")
    print(f"{'#' * 70}", flush=True)
    t0 = time.time()
    fp = build_field_problem(nx=nx, ny=nx,
                             ell_c=ell_c,
                             sigma_delta_ratio=sigma_delta_ratio,
                             energy_tol=energy_tol)
    print(f"  Built field problem: n_dim = {fp.n_dim}", flush=True)
    try:
        res = build_pce_field(order, fp, quad_order=quad_order, verbose=True)
        elapsed = time.time() - t0
        print(f"\n  PCE {nx}x{nx} p={order}, Q={q} took {elapsed:.1f}s",
              flush=True)
        return {
            "kind": "PCE", "nx": nx, "ny": nx, "order": order,
            "quad_order": res.quad_order,
            "n_dim": fp.n_dim, "n_basis": res.n_basis, "n_quad": res.n_quad,
            "coeffs_E": res.coeffs_E, "coeffs_nu": res.coeffs_nu,
            "E_star_model": res.E_star_model,
            "nu_star_model": res.nu_star_model,
            "mean_E":  res.mean_E, "var_E":  res.var_E,
            "mean_nu": res.mean_nu, "var_nu": res.var_nu,
            "elapsed_sec": elapsed,
        }
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  PCE {nx}x{nx} p={order}, Q={q} FAILED "
              f"after {elapsed:.1f}s: {e}", flush=True)
        return {
            "kind": "PCE_failed", "nx": nx, "ny": nx, "order": order,
            "quad_order": q, "n_dim": fp.n_dim,
            "error": str(e),
            "elapsed_sec": elapsed,
        }

DEFAULT_RVES = [2, 3, 4, 5]
DEFAULT_PCE_ORDERS = [2, 3]
DEFAULT_INNER_RVE = 2
DEFAULT_INNER_QUAD_ORDERS = [2, 3, 4, 5]


def default_jobs(N_mc=100000, rves=None, pce_orders=None, quad_orders=None):
    if rves is None:
        rves = DEFAULT_RVES
    if pce_orders is None:
        pce_orders = DEFAULT_PCE_ORDERS
    if quad_orders is None:
        quad_orders = DEFAULT_INNER_QUAD_ORDERS

    jobs = []
    for nx in rves:
        jobs.append(("MC", nx, N_mc))
        if nx == DEFAULT_INNER_RVE:
            for order in pce_orders:
                jobs.extend(("PCE", nx, order, q) for q in quad_orders)
        elif nx == 3:
            if 2 in pce_orders:
                jobs.append(("PCE", nx, 2, 2))
            if 3 in pce_orders:
                jobs.append(("PCE", nx, 3, 3))
        else:
            if 2 in pce_orders:
                jobs.append(("PCE", nx, 2, 2))
    return jobs


def quadrature_jobs(rves=None, pce_orders=None, quad_orders=None, N_mc=100000):
    if rves is None:
        rves = DEFAULT_RVES
    if pce_orders is None:
        pce_orders = DEFAULT_PCE_ORDERS
    if quad_orders is None:
        quad_orders = DEFAULT_INNER_QUAD_ORDERS
    jobs = []
    for nx in rves:
        jobs.append(("MC", nx, N_mc))
        for order in pce_orders:
            if nx in (4, 5) and order == 3:
                continue
            jobs.extend(("PCE", nx, order, q) for q in quad_orders)
    return jobs


def job_key(job):
    kind, nx, arg = job[:3]
    if kind == "MC":
        return f"mc-{nx}x{nx}"
    if len(job) >= 4 and job[3] is not None:
        return f"{nx}x{nx}-p{arg}-q{job[3]}"
    return f"{nx}x{nx}-p{arg}"

def main():
    parser = argparse.ArgumentParser(description="RVE x PCE-order sweep.")
    parser.add_argument("--out", type=str, default="results_sweep.pkl",
                        help="output pickle filename")
    parser.add_argument("--N", type=int, default=100000,
                        help="MC sample count (default 100000)")
    parser.add_argument("--mc-only", action="store_true",
                        help="only run MC jobs")
    parser.add_argument("--pce-only", action="store_true",
                        help="only run PCE jobs")
    parser.add_argument("--skip-heavy", action="store_true",
                        help="compatibility flag; 4x4/5x5 p=3 are excluded "
                             "from the default plan already")
    parser.add_argument("--quad-sweep", action="store_true",
                        help="legacy alias; combined p and Q sweep is now default")
    parser.add_argument("--rves", type=int, nargs="+", default=DEFAULT_RVES,
                        help="RVE sizes to run, in order (default 2 3 4 5)")
    parser.add_argument("--quad-rve", type=int, default=None,
                        help="single RVE size for --quad-sweep; use --quad-rves "
                             "for multiple sizes")
    parser.add_argument("--quad-rves", type=int, nargs="+", default=None,
                        help="RVE sizes for --quad-sweep, in order "
                             "(default follows --rves)")
    parser.add_argument("--pce-order", type=int, default=None,
                        help="single PCE order p to run; overrides --pce-orders")
    parser.add_argument("--pce-orders", type=int, nargs="+",
                        default=DEFAULT_PCE_ORDERS,
                        help="PCE orders p to run, in order (default 2 3)")
    parser.add_argument("--quad-orders", type=int, nargs="+",
                        default=DEFAULT_INNER_QUAD_ORDERS,
                        help="inner-loop quadrature levels Q for 2x2 "
                             "(default 2 3 4 5)")
    parser.add_argument("--only", type=str, nargs="+", default=None,
                        help="run only jobs matching these keys "
                             "(e.g. --only 2x2-p3-q4 3x3-p2-q2 mc-4x4)")
    args = parser.parse_args()

    if args.quad_sweep:
        quad_rves = args.quad_rves
        if quad_rves is None:
            quad_rves = [args.quad_rve] if args.quad_rve is not None else args.rves
        pce_orders = [args.pce_order] if args.pce_order is not None else args.pce_orders
        all_jobs = quadrature_jobs(rves=quad_rves, pce_orders=pce_orders,
                                   quad_orders=args.quad_orders, N_mc=args.N)
    else:
        pce_orders = [args.pce_order] if args.pce_order is not None else args.pce_orders
        all_jobs = default_jobs(N_mc=args.N, rves=args.rves,
                                pce_orders=pce_orders,
                                quad_orders=args.quad_orders)

    if args.mc_only:
        all_jobs = [j for j in all_jobs if j[0] == "MC"]
    if args.pce_only:
        all_jobs = [j for j in all_jobs if j[0] == "PCE"]
    if args.skip_heavy:
        all_jobs = [j for j in all_jobs
                    if not (j[0] == "PCE" and j[1] in (4, 5) and j[2] == 3)]
    if args.only:
        all_jobs = [j for j in all_jobs if job_key(j) in args.only]

    if not all_jobs:
        print("No jobs match the filters.")
        sys.exit(1)

    print("=" * 70)
    print("RANDOM-FIELD UQ CLUSTER SWEEP")
    print("=" * 70)
    print(f"Output: {args.out}")
    print(f"Jobs ({len(all_jobs)}):")
    for job in all_jobs:
        kind, nx, arg = job[:3]
        if kind == "MC":
            print(f"   MC  {nx}x{nx}  N={arg}")
        else:
            q = job[3] if len(job) >= 4 else arg
            print(f"   PCE {nx}x{nx}  order={arg}  Q={q}   [{job_key(job)}]")
    print("=" * 70, flush=True)

    results = []
    total_t0 = time.time()

    for i, job in enumerate(all_jobs):
        kind, nx, arg = job[:3]
        try:
            if kind == "MC":
                r = run_mc_for_rve(nx, N=arg)
            else:
                quad_order = job[3] if len(job) >= 4 else None
                r = run_pce_for_rve(nx, order=arg, quad_order=quad_order)
            results.append(r)
        except Exception as e:
            print(f"\n!!! JOB {kind} {nx}x{nx} CRASHED: {e}", flush=True)
            results.append({
                "kind": kind + "_crash", "nx": nx, "arg": arg,
                "error": str(e),
            })

        with open(args.out, "wb") as f:
            pickle.dump(results, f)
        print(f"\n[saved {len(results)}/{len(all_jobs)} jobs to {args.out}]",
              flush=True)
        print(f"[cumulative wall time: {(time.time()-total_t0)/60:.1f} min]",
              flush=True)

    total_elapsed = time.time() - total_t0
    print("\n" + "=" * 70)
    print(f"SWEEP COMPLETE — total wall time: {total_elapsed/60:.1f} min")
    print("=" * 70)
    print()
    print(f"  {'kind':>12} {'RVE':>5} {'dim':>5} {'param':>7} "
          f"{'mean E*':>14} {'Var E*':>14} {'mean nu*':>10} {'time (s)':>10}")
    print("-" * 100)
    for r in results:
        k = r.get("kind", "?")
        if k.endswith("_crash") or k == "PCE_failed":
            err_msg = (r.get("error") or "?")[:40]
            print(f"  {k:>12} {r.get('nx','?'):>2}x{r.get('nx','?'):<2} "
                  f"   --       --             --             --        "
                  f"--   (error: {err_msg})")
            continue
        tag = f"PCE p={r['order']},Q={r.get('quad_order', r['order'])}"\
              if r["kind"] == "PCE"\
              else f"MC N={r.get('N','?')}"
        param = r.get("order", r.get("N"))
        print(f"  {tag:>12} {r['nx']:>2}x{r['nx']:<2} "
              f"{r['n_dim']:>5d} {str(param):>7} "
              f"{r['mean_E']:>14.6e} {r['var_E']:>14.6e} "
              f"{r['mean_nu']:>10.4f} {r['elapsed_sec']:>10.1f}")

    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()

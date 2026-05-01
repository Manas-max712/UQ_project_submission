from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional
import time
import sys

from monte_carlo import InputSpec
from forward_map import build_field_problem, forward_map, FieldProblem


@dataclass
class MCFieldResult:
    xi_samples: np.ndarray
    E_star:    np.ndarray
    nu_star:   np.ndarray
    n_failures: int
    elapsed_sec: float
    n_dim:      int
    spec:       InputSpec
    ell_c:      float
    sigma_delta_ratio: float

    def stats(self, field_name: str = "E_star") -> dict:
        vals = getattr(self, field_name)
        ok = np.isfinite(vals)
        v = vals[ok]
        return {
            "N":       int(ok.sum()),
            "mean":    float(np.mean(v)),
            "std":     float(np.std(v, ddof=1)),
            "CoV":     float(np.std(v, ddof=1) / np.abs(np.mean(v))),
            "min":     float(np.min(v)),
            "max":     float(np.max(v)),
            "median":  float(np.median(v)),
            "p05":     float(np.percentile(v, 5)),
            "p95":     float(np.percentile(v, 95)),
            "SE_mean": float(np.std(v, ddof=1) / np.sqrt(len(v))),
        }

    def running(self, field_name: str = "E_star", stride: int = 100) -> tuple:
        v = getattr(self, field_name)
        v = v[np.isfinite(v)]
        N = len(v)
        idx = np.arange(stride, N + 1, stride)
        run_mean = np.array([np.mean(v[:n]) for n in idx])
        run_std  = np.array([np.std(v[:n], ddof=1) for n in idx])
        return idx, run_mean, run_std

def run_mc_field(N: int,
                 fp: Optional[FieldProblem] = None,
                 spec: Optional[InputSpec] = None,
                 nx: int = 2, ny: int = 2,
                 ell_c: float = 1.5,
                 sigma_delta_ratio: float = 0.03,
                 energy_tol: float = 0.95,
                 seed: int = 2026,
                 verbose: bool = True,
                 progress_every: Optional[int] = None) -> MCFieldResult:
    if fp is None:
        if spec is None:
            spec = InputSpec()
        fp = build_field_problem(spec=spec, nx=nx, ny=ny,
                                 ell_c=ell_c,
                                 sigma_delta_ratio=sigma_delta_ratio,
                                 energy_tol=energy_tol)

    n_dim = fp.n_dim
    if progress_every is None:
        progress_every = max(1, N // 10)

    rng = np.random.default_rng(seed)
    xi_samples = rng.standard_normal((N, n_dim))

    E_star  = np.full(N, np.nan)
    nu_star = np.full(N, np.nan)
    n_fail  = 0

    if verbose:
        print("=" * 70)
        print(f"Random-field MC  —  N = {N:,d}, dim = {n_dim}, "
              f"RVE = {nx}x{ny}, seed = {seed}")
        print("=" * 70)
        print(f"  correlation length ell_c       : {ell_c}")
        print(f"  node perturbation sigma / l    : {sigma_delta_ratio}")
        print(f"  input spec: {fp.spec.describe().strip()}")
        print()

    t0 = time.time()
    for i in range(N):
        xi_i = xi_samples[i]
        try:
            E_eff, nu_eff = forward_map(xi_i, fp)
            if np.isfinite(E_eff) and np.isfinite(nu_eff):
                E_star[i]  = E_eff
                nu_star[i] = nu_eff
            else:
                n_fail += 1
        except Exception as e:
            n_fail += 1
            if verbose and n_fail <= 3:
                print(f"  [warn] sample {i} failed: {e}", file=sys.stderr)

        if verbose and (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta  = (N - i - 1) / rate
            vv = E_star[:i+1]
            ok = np.isfinite(vv)
            if ok.sum() > 1:
                m  = np.mean(vv[ok])
                se = np.std(vv[ok], ddof=1) / np.sqrt(ok.sum())
                print(f"  {i+1:>7,d}/{N:<7,d} "
                      f"({(i+1)/N*100:5.1f}%)  "
                      f"rate={rate:6.0f} samples/s  "
                      f"eta={eta:6.1f}s   "
                      f"E*: {m:.4e} ± {se:.2e}")

    elapsed = time.time() - t0
    if verbose:
        print()
        print(f"Done in {elapsed:.2f} s ({N/elapsed:.0f} samples/s), "
              f"{n_fail} failures ({n_fail/max(N,1)*100:.2f}%)")

    return MCFieldResult(xi_samples=xi_samples,
                         E_star=E_star, nu_star=nu_star,
                         n_failures=n_fail, elapsed_sec=elapsed,
                         n_dim=n_dim, spec=fp.spec,
                         ell_c=ell_c,
                         sigma_delta_ratio=sigma_delta_ratio)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Random-field MC UQ for honeycomb.")
    parser.add_argument("-N", type=int, default=10000,
                        help="number of MC samples (default 10000)")
    parser.add_argument("--nx", type=int, default=2)
    parser.add_argument("--ny", type=int, default=2)
    parser.add_argument("--ell-c", type=float, default=1.5)
    parser.add_argument("--sigma-delta", type=float, default=0.03,
                        help="node-perturbation std / l_nominal (default 0.03)")
    parser.add_argument("--energy-tol", type=float, default=0.95,
                        help="K-L truncation energy threshold (default 0.95)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save-npz", type=str, default=None)
    args = parser.parse_args()

    result = run_mc_field(N=args.N, nx=args.nx, ny=args.ny,
                          ell_c=args.ell_c,
                          sigma_delta_ratio=args.sigma_delta,
                          energy_tol=args.energy_tol,
                          seed=args.seed)

    print()
    print("-" * 70)
    print("Output statistics:")
    print("-" * 70)
    for qoi in ("E_star", "nu_star"):
        s = result.stats(qoi)
        print(f"\n  {qoi}:")
        print(f"    N valid    = {s['N']:,d}")
        print(f"    mean       = {s['mean']:.6e}")
        print(f"    std        = {s['std']:.6e}")
        print(f"    CoV        = {s['CoV']*100:.2f} %")
        print(f"    5%–95%     = [{s['p05']:.4e}, {s['p95']:.4e}]")
        print(f"    min / max  = [{s['min']:.4e}, {s['max']:.4e}]")
        print(f"    SE(mean)   = {s['SE_mean']:.3e}  "
              f"(relative: {s['SE_mean']/abs(s['mean'])*100:.3f} %)")

    if args.save_npz is not None:
        np.savez_compressed(args.save_npz,
                            xi_samples=result.xi_samples,
                            E_star=result.E_star,
                            nu_star=result.nu_star,
                            elapsed_sec=result.elapsed_sec,
                            n_failures=result.n_failures,
                            n_dim=result.n_dim,
                            ell_c=result.ell_c,
                            sigma_delta_ratio=result.sigma_delta_ratio)
        print(f"\n  results saved to {args.save_npz}")

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable
import time
import sys

from honeycomb_fe import forward_solve, effective_stiffness, effective_properties, build_honeycomb_rve


@dataclass
class InputSpec:

    theta_mean_deg: float = 30.0
    theta_std_deg:  float = 0.9

    t_mean: float = 0.05
    t_cov:  float = 0.10

    l_mean: float = 1.0
    l_cov:  float = 0.03

    E_mean: float = 200.0
    E_cov:  float = 0.10

    def lognormal_params(self, mean: float, cov: float) -> tuple:

        sigma_log2 = np.log(1.0 + cov * cov)
        sigma_log  = np.sqrt(sigma_log2)
        mu_log     = np.log(mean) - 0.5 * sigma_log2
        return mu_log, sigma_log

    def sample(self, N: int, rng: np.random.Generator) -> np.ndarray:

        theta = rng.normal(self.theta_mean_deg, self.theta_std_deg, size=N)

        mu, sg = self.lognormal_params(self.t_mean, self.t_cov)
        t = rng.lognormal(mean=mu, sigma=sg, size=N)

        mu, sg = self.lognormal_params(self.l_mean, self.l_cov)
        l = rng.lognormal(mean=mu, sigma=sg, size=N)

        mu, sg = self.lognormal_params(self.E_mean, self.E_cov)
        E = rng.lognormal(mean=mu, sigma=sg, size=N)

        return np.column_stack([theta, t, l, E])

    def describe(self) -> str:
        lines = [
            f"  theta ~ Normal({self.theta_mean_deg}°, {self.theta_std_deg}°)   "
                f"CoV ≈ {self.theta_std_deg/self.theta_mean_deg*100:.1f}%",
            f"  t     ~ Lognormal(mean={self.t_mean}, CoV={self.t_cov*100:.1f}%)",
            f"  l     ~ Lognormal(mean={self.l_mean}, CoV={self.l_cov*100:.1f}%)",
            f"  E     ~ Lognormal(mean={self.E_mean}, CoV={self.E_cov*100:.1f}%)",
        ]
        return "\n".join(lines)


@dataclass
class MCResult:


    samples: np.ndarray


    E_star: np.ndarray
    nu_star: np.ndarray


    C_star: Optional[np.ndarray]


    n_failures: int
    elapsed_sec: float


    spec: InputSpec


    def stats(self, field_name: str = "E_star") -> dict:

        vals = getattr(self, field_name)
        ok = np.isfinite(vals)
        v  = vals[ok]
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


    def running(self, field_name: str = "E_star", stride: int = 1) -> tuple:

        v = getattr(self, field_name)

        v = v[np.isfinite(v)]
        N = len(v)
        idx = np.arange(stride, N + 1, stride)
        run_mean = np.array([np.mean(v[:n]) for n in idx])
        run_std  = np.array([np.std(v[:n], ddof=1) if n > 1 else 0.0 for n in idx])
        return idx, run_mean, run_std


def run_monte_carlo(N: int,
                    spec: Optional[InputSpec] = None,
                    nx: int = 2, ny: int = 2,
                    seed: int = 2026,
                    save_C: bool = False,
                    verbose: bool = True,
                    progress_every: Optional[int] = None) -> MCResult:

    if spec is None:
        spec = InputSpec()
    if progress_every is None:
        progress_every = max(1, N // 10)

    rng = np.random.default_rng(seed)
    samples = spec.sample(N, rng)

    E_star  = np.full(N, np.nan)
    nu_star = np.full(N, np.nan)
    C_all   = np.full((N, 3, 3), np.nan) if save_C else None
    n_fail  = 0

    if verbose:
        print("=" * 68)
        print(f"Monte Carlo forward UQ  —  N = {N:,d}, RVE = {nx}x{ny}, seed = {seed}")
        print("=" * 68)
        print("Input distributions:")
        print(spec.describe())
        print()

    t0 = time.time()
    for i in range(N):
        theta, t, l, E = samples[i]


        if not (1.0 < theta < 89.0):
            n_fail += 1
            continue

        try:
            if save_C:
                E_eff, nu_eff, C, _ = forward_solve(theta, l, t, E, nx, ny,
                                                   return_full=True)
                C_all[i] = C
            else:
                E_eff, nu_eff = forward_solve(theta, l, t, E, nx, ny)

            if not (np.isfinite(E_eff) and np.isfinite(nu_eff)):
                n_fail += 1
                continue

            E_star[i]  = E_eff
            nu_star[i] = nu_eff

        except Exception as e:
            n_fail += 1
            if verbose and n_fail <= 3:
                print(f"  [warn] sample {i} failed: {e}", file=sys.stderr)

        if verbose and (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (N - i - 1) / rate

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
              f"{n_fail} failures ({n_fail/N*100:.2f}%)")

    return MCResult(samples=samples, E_star=E_star, nu_star=nu_star,
                    C_star=C_all, n_failures=n_fail, elapsed_sec=elapsed,
                    spec=spec)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MC forward UQ for honeycomb.")
    parser.add_argument("-N", type=int, default=10000,
                        help="number of MC samples (default 10000)")
    parser.add_argument("--nx", type=int, default=2)
    parser.add_argument("--ny", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save-C", action="store_true",
                        help="also store the full 3x3 C* for each sample")
    parser.add_argument("--save-npz", type=str, default=None,
                        help="save results to .npz file at this path")
    args = parser.parse_args()

    result = run_monte_carlo(N=args.N, nx=args.nx, ny=args.ny,
                             seed=args.seed, save_C=args.save_C)


    print()
    print("-" * 68)
    print("Output statistics:")
    print("-" * 68)
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
        to_save = {
            "samples": result.samples,
            "E_star":  result.E_star,
            "nu_star": result.nu_star,
            "elapsed_sec": result.elapsed_sec,
            "n_failures":  result.n_failures,
        }
        if result.C_star is not None:
            to_save["C_star"] = result.C_star
        np.savez_compressed(args.save_npz, **to_save)
        print(f"\n  results saved to {args.save_npz}")

from __future__ import annotations
import numpy as np
import chaospy as cp
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time
import warnings

from forward_map import FieldProblem, build_field_problem, forward_map
from hermite_fast import (
    gen_multiindices, compute_norms, project_fast, eval_surrogate_fast,
)

warnings.filterwarnings("ignore")


class HermiteSurrogate:
    def __init__(self, coeffs: np.ndarray, multi_indices: np.ndarray,
                 norms: np.ndarray, order: int):
        self._coeffs = coeffs
        self._mi = multi_indices
        self._norms = norms
        self._p = order

    def __call__(self, *xi_args) -> np.ndarray:
        xi = np.stack([np.atleast_1d(x) for x in xi_args], axis=0)
        return eval_surrogate_fast(xi, self._coeffs, self._mi,
                                   self._norms, self._p)


def evaluate_fwd_map_at_points(xi_grid: np.ndarray,
                               fp: FieldProblem,
                               verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    n_dim, M = xi_grid.shape
    assert n_dim == fp.n_dim

    E_star = np.zeros(M)
    nu_star = np.zeros(M)
    n_fail = 0
    t0 = time.time()
    for k in range(M):
        try:
            E_k, nu_k = forward_map(xi_grid[:, k], fp)
            if np.isfinite(E_k) and np.isfinite(nu_k):
                E_star[k] = E_k
                nu_star[k] = nu_k
            else:
                E_star[k] = np.nan
                nu_star[k] = np.nan
                n_fail += 1
        except Exception:
            E_star[k] = np.nan
            nu_star[k] = np.nan
            n_fail += 1
        if verbose and M >= 1000 and (k + 1) % max(1, M // 10) == 0:
            elapsed = time.time() - t0
            rate = (k + 1) / elapsed
            eta = (M - k - 1) / rate
            print(f"    {k+1:>6d}/{M:<6d} ({(k+1)/M*100:5.1f}%)  "
                  f"rate = {rate:5.0f}/s  eta = {eta:6.1f}s")
    elapsed = time.time() - t0
    if verbose:
        print(f"  evaluated {M} points in {elapsed:.1f} s "
              f"({M/elapsed:.0f} pts/s, {n_fail} failures)")
    return E_star, nu_star


@dataclass
class PCEFieldResult:
    order: int
    quad_order: int
    n_basis: int
    n_quad: int
    n_dim: int
    coeffs_E: np.ndarray
    coeffs_nu: np.ndarray
    surrogate_E: object
    surrogate_nu: object
    xi_grid: np.ndarray
    weights: np.ndarray
    E_star_model: np.ndarray
    nu_star_model: np.ndarray
    mean_E: float
    var_E:  float
    mean_nu: float
    var_nu: float
    elapsed_sec: float


def build_pce_field(order: int, fp: FieldProblem,
                    quad_order: Optional[int] = None,
                    sparse: bool = True, verbose: bool = True) -> PCEFieldResult:
    n_dim = fp.n_dim
    if quad_order is None:
        quad_order = order
    t0_total = time.time()

    xi_dist = cp.J(*[cp.Normal(0, 1) for _ in range(n_dim)])
    xi_grid, weights = cp.generate_quadrature(quad_order, xi_dist,
                                              rule="gaussian", sparse=sparse)
    n_quad = xi_grid.shape[1]

    multi_indices = gen_multiindices(n_dim, order)
    norms = compute_norms(multi_indices)
    n_basis = multi_indices.shape[0]

    if verbose:
        print(f"\n=== PCE order p = {order}, quad Q = {quad_order}, dim = {n_dim} ===")
        print(f"  basis size: {n_basis}")
        print(f"  quad points: {n_quad}")

    E_star, nu_star = evaluate_fwd_map_at_points(xi_grid, fp, verbose=verbose)

    if verbose:
        print("  fast projection for E*...")
    coeffs_E_arr = project_fast(xi_grid, weights, E_star,
                                multi_indices, norms, order,
                                batch_size=4000, verbose=verbose)
    if verbose:
        print("  fast projection for nu*...")
    coeffs_nu_arr = project_fast(xi_grid, weights, nu_star,
                                 multi_indices, norms, order,
                                 batch_size=4000, verbose=verbose)

    surrogate_E  = HermiteSurrogate(coeffs_E_arr, multi_indices, norms, order)
    surrogate_nu = HermiteSurrogate(coeffs_nu_arr, multi_indices, norms, order)

    mean_E  = float(coeffs_E_arr[0])
    var_E   = float(np.sum(coeffs_E_arr[1:]**2))
    mean_nu = float(coeffs_nu_arr[0])
    var_nu  = float(np.sum(coeffs_nu_arr[1:]**2))

    elapsed = time.time() - t0_total
    if verbose:
        print(f"  E[E*]   = {mean_E:.6e}, Var[E*]   = {var_E:.6e}")
        print(f"  E[nu*]  = {mean_nu:.6e}, Var[nu*]  = {var_nu:.6e}")
        print(f"  total elapsed: {elapsed:.1f} s")
    return PCEFieldResult(
        order=order, quad_order=quad_order,
        n_basis=n_basis, n_quad=n_quad, n_dim=n_dim,
        coeffs_E=coeffs_E_arr, coeffs_nu=coeffs_nu_arr,
        surrogate_E=surrogate_E, surrogate_nu=surrogate_nu,
        xi_grid=xi_grid, weights=weights,
        E_star_model=E_star, nu_star_model=nu_star,
        mean_E=mean_E, var_E=var_E, mean_nu=mean_nu, var_nu=var_nu,
        elapsed_sec=elapsed)


def convergence_study_field(orders: List[int],
                            fp: Optional[FieldProblem] = None,
                            quad_order: Optional[int] = None,
                            **fp_kwargs) -> List[PCEFieldResult]:
    if fp is None:
        fp = build_field_problem(**fp_kwargs)
    print("=" * 70)
    print(f"Random-field PCE convergence study — orders {orders}, dim = {fp.n_dim}")
    print("=" * 70)
    print(fp.kl_model.summary())
    print()
    results = []
    for p in orders:
        results.append(build_pce_field(p, fp, quad_order=quad_order, verbose=True))
    print()
    print("-" * 80)
    print("Convergence summary:")
    print("-" * 80)
    print(f"  {'order':>6} {'Q':>4} {'Nbasis':>8} {'Nquad':>8} "
          f"{'E[E*]':>14} {'Var[E*]':>14} "
          f"{'E[nu*]':>10} {'Var[nu*]':>12}")
    for r in results:
        print(f"  {r.order:>6d} {r.quad_order:>4d} "
              f"{r.n_basis:>8d} {r.n_quad:>8d} "
              f"{r.mean_E:>14.6e} {r.var_E:>14.6e} "
              f"{r.mean_nu:>10.4f} {r.var_nu:>12.4e}")
    return results


def quadrature_convergence_study_field(order: int,
                                       quad_orders: List[int],
                                       fp: Optional[FieldProblem] = None,
                                       **fp_kwargs) -> List[PCEFieldResult]:
    if fp is None:
        fp = build_field_problem(**fp_kwargs)
    print("=" * 70)
    print(f"Random-field PCE quadrature study — p = {order}, "
          f"Q = {quad_orders}, dim = {fp.n_dim}")
    print("=" * 70)
    print(fp.kl_model.summary())
    print()

    results = []
    for q in quad_orders:
        results.append(build_pce_field(order, fp, quad_order=q, verbose=True))

    print()
    print("-" * 86)
    print("Quadrature convergence summary:")
    print("-" * 86)
    print(f"  {'p':>4} {'Q':>4} {'Nbasis':>8} {'Nquad':>8} "
          f"{'E[E*]':>14} {'Var[E*]':>14} "
          f"{'E[nu*]':>10} {'Var[nu*]':>12}")
    for r in results:
        print(f"  {r.order:>4d} {r.quad_order:>4d} "
              f"{r.n_basis:>8d} {r.n_quad:>8d} "
              f"{r.mean_E:>14.6e} {r.var_E:>14.6e} "
              f"{r.mean_nu:>10.4f} {r.var_nu:>12.4e}")
    return results


def sample_surrogate_field(result: PCEFieldResult, N: int = 100000,
                           seed: int = 2026,
                           expansion=None) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xi_samples = rng.standard_normal((result.n_dim, N))
    E_samples  = np.asarray(result.surrogate_E(*xi_samples)).ravel()
    nu_samples = np.asarray(result.surrogate_nu(*xi_samples)).ravel()
    return E_samples, nu_samples


if __name__ == "__main__":
    import pickle
    fp = build_field_problem()
    print(f"Stochastic dimension: {fp.n_dim}")
    orders = [2, 3]
    results = convergence_study_field(orders, fp=fp)
    with open("/home/claude/honeycomb_fe/pce_field_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\nResults saved to pce_field_results.pkl")

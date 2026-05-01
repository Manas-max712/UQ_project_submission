from __future__ import annotations
import numpy as np
import chaospy as cp
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import time

from honeycomb_fe import forward_solve
from monte_carlo import InputSpec


def xi_to_physical(xi: np.ndarray, spec: InputSpec) -> Tuple[float, float, float, float]:

    xi1, xi2, xi3, xi4 = xi

    theta = spec.theta_mean_deg + spec.theta_std_deg * xi1

    mu, sg = spec.lognormal_params(spec.t_mean, spec.t_cov)
    t = np.exp(mu + sg * xi2)

    mu, sg = spec.lognormal_params(spec.l_mean, spec.l_cov)
    l = np.exp(mu + sg * xi3)

    mu, sg = spec.lognormal_params(spec.E_mean, spec.E_cov)
    E = np.exp(mu + sg * xi4)

    return theta, t, l, E


def evaluate_model_at_points(xi_grid: np.ndarray, spec: InputSpec,
                             nx: int = 2, ny: int = 2,
                             verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:

    M = xi_grid.shape[1]
    E_star = np.zeros(M)
    nu_star = np.zeros(M)

    t0 = time.time()
    for k in range(M):
        theta, t, l, E = xi_to_physical(xi_grid[:, k], spec)
        try:
            E_eff, nu_eff = forward_solve(theta, l, t, E, nx, ny)
            E_star[k] = E_eff
            nu_star[k] = nu_eff
        except Exception as e:
            if verbose:
                print(f"  [warn] point {k} failed: {e}")
            E_star[k] = np.nan
            nu_star[k] = np.nan
    elapsed = time.time() - t0
    if verbose:
        print(f"  evaluated {M} points in {elapsed:.2f} s "
              f"({M/elapsed:.0f} pts/s)")

    return E_star, nu_star


@dataclass
class PCEResult:

    order: int
    quadrature_order: int
    n_basis: int
    n_quad: int
    coeffs_E: np.ndarray
    coeffs_nu: np.ndarray
    surrogate_E: object
    surrogate_nu: object
    xi_grid: np.ndarray
    weights: np.ndarray
    E_star_model: np.ndarray
    nu_star_model: np.ndarray
    elapsed_sec: float


    mean_E: float
    var_E: float
    mean_nu: float
    var_nu: float


def build_pce(order: int, spec: InputSpec,
              nx: int = 2, ny: int = 2,
              quadrature_order: Optional[int] = None,
              quadrature_rule: str = "gaussian",
              sparse: bool = True,
              precomputed_quad: Optional[Tuple[np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray]] = None,
              verbose: bool = True) -> PCEResult:

    if quadrature_order is None:
        quadrature_order = order


    xi_dist = cp.J(cp.Normal(0, 1), cp.Normal(0, 1),
                   cp.Normal(0, 1), cp.Normal(0, 1))


    expansion, norms = cp.generate_expansion(order, xi_dist,
                                             normed=True, retall=True)
    n_basis = len(expansion)


    if precomputed_quad is None:
        xi_grid, weights = cp.generate_quadrature(quadrature_order, xi_dist,
                                                  rule=quadrature_rule,
                                                  sparse=sparse)
        E_star = None
        nu_star = None
    else:
        xi_grid, weights, E_star, nu_star = precomputed_quad
    n_quad = xi_grid.shape[1]

    if verbose:
        print(f"\n=== PCE order p = {order}, quadrature depth Q = {quadrature_order} ===")
        print(f"  basis size:    {n_basis}")
        print(f"  quad points:   {n_quad}")

    t0 = time.time()
    if E_star is None or nu_star is None:

        E_star, nu_star = evaluate_model_at_points(xi_grid, spec, nx, ny, verbose)
    elif verbose:
        print("  reusing cached model evaluations for this quadrature grid")
    elapsed = time.time() - t0


    surrogate_E, coeffs_E = cp.fit_quadrature(expansion, xi_grid, weights,
                                              E_star, retall=True,
                                              norms=norms)
    surrogate_nu, coeffs_nu = cp.fit_quadrature(expansion, xi_grid, weights,
                                                nu_star, retall=True,
                                                norms=norms)


    mean_E = float(cp.E(surrogate_E, xi_dist))
    var_E  = float(cp.Var(surrogate_E, xi_dist))
    mean_nu = float(cp.E(surrogate_nu, xi_dist))
    var_nu  = float(cp.Var(surrogate_nu, xi_dist))

    if verbose:
        print(f"  E[E*]   = {mean_E:.6e}, Var[E*]   = {var_E:.6e}")
        print(f"  E[nu*]  = {mean_nu:.6e}, Var[nu*]  = {var_nu:.6e}")

    return PCEResult(
        order=order, quadrature_order=quadrature_order,
        n_basis=n_basis, n_quad=n_quad,
        coeffs_E=np.asarray(coeffs_E), coeffs_nu=np.asarray(coeffs_nu),
        surrogate_E=surrogate_E, surrogate_nu=surrogate_nu,
        xi_grid=xi_grid, weights=weights,
        E_star_model=E_star, nu_star_model=nu_star,
        elapsed_sec=elapsed,
        mean_E=mean_E, var_E=var_E,
        mean_nu=mean_nu, var_nu=var_nu,
    )


def convergence_study(orders: List[int], spec: Optional[InputSpec] = None,
                      nx: int = 2, ny: int = 2,
                      verbose: bool = True) -> List[PCEResult]:

    if spec is None:
        spec = InputSpec()

    if verbose:
        print("=" * 68)
        print(f"PCE convergence study — orders {orders}, RVE {nx}x{ny}")
        print("=" * 68)
        print("Input distributions:")
        print(spec.describe())

    results = []
    for p in orders:
        r = build_pce(p, spec, nx, ny, verbose=verbose)
        results.append(r)

    if verbose:
        print()
        print("-" * 72)
        print("Convergence summary:")
        print("-" * 72)
        print(f"  {'order':>6} {'Nbasis':>8} {'Nquad':>8} "
              f"{'E[E*]':>14} {'Var[E*]':>14} "
              f"{'E[nu*]':>10} {'Var[nu*]':>12}")
        for r in results:
            print(f"  {r.order:>6d} {r.n_basis:>8d} {r.n_quad:>8d} "
                  f"{r.mean_E:>14.6e} {r.var_E:>14.6e} "
                  f"{r.mean_nu:>10.4f} {r.var_nu:>12.4e}")

    return results


def sample_surrogate(result: PCEResult, N: int = 100000,
                     seed: int = 2026) -> Tuple[np.ndarray, np.ndarray]:

    xi_dist = cp.J(cp.Normal(0, 1), cp.Normal(0, 1),
                   cp.Normal(0, 1), cp.Normal(0, 1))
    rng = np.random.default_rng(seed)

    xi_samples = rng.standard_normal((4, N))
    E_samples  = np.asarray(result.surrogate_E(*xi_samples)).ravel()
    nu_samples = np.asarray(result.surrogate_nu(*xi_samples)).ravel()
    return E_samples, nu_samples


if __name__ == "__main__":
    orders = [2, 3, 4, 5, 6]
    results = convergence_study(orders)


    import pickle
    with open("/home/claude/honeycomb_fe/pce_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\nPCE results saved to pce_results.pkl")

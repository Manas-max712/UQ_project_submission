from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Callable
import time

from honeycomb_fe import build_honeycomb_rve, Mesh
from random_field import KLModel
from mesh_kl import build_kl_model, expand_node_perturbations, master_node_ids
from heterogeneous_solve import forward_solve_heterogeneous
from monte_carlo import InputSpec


@dataclass
class FieldProblem:
    mesh: Mesh
    kl_model: KLModel
    spec: InputSpec
    master_ids: list
    mu_L_t: float = 0.0
    sigma_L_t: float = 0.0
    mu_L_E: float = 0.0
    sigma_L_E: float = 0.0

    @property
    def n_dim(self) -> int:
        return self.kl_model.n_dim


def build_field_problem(spec: Optional[InputSpec] = None,
                        nx: int = 2, ny: int = 2,
                        ell_c: float = 1.5,
                        sigma_delta_ratio: float = 0.03,
                        energy_tol: float = 0.95) -> FieldProblem:
    if spec is None:
        spec = InputSpec()

    mesh = build_honeycomb_rve(spec.theta_mean_deg, spec.l_mean, nx, ny)
    kl_model = build_kl_model(mesh,
                              ell_c=ell_c,
                              sigma_delta_ratio=sigma_delta_ratio,
                              l_nominal=spec.l_mean,
                              energy_tol=energy_tol)
    m_ids = master_node_ids(mesh)

    mu_L_t,   sigma_L_t   = spec.lognormal_params(spec.t_mean, spec.t_cov)
    mu_L_E,   sigma_L_E   = spec.lognormal_params(spec.E_mean, spec.E_cov)

    return FieldProblem(mesh=mesh, kl_model=kl_model, spec=spec,
                        master_ids=m_ids,
                        mu_L_t=mu_L_t, sigma_L_t=sigma_L_t,
                        mu_L_E=mu_L_E, sigma_L_E=sigma_L_E)

def forward_map(xi: np.ndarray, fp: FieldProblem,
                return_full: bool = False):
    dx_master, dy_master, g_t_mid, g_E_mid = fp.kl_model.reconstruct_all(xi)

    delta_master = np.column_stack([dx_master, dy_master])
    full_delta = expand_node_perturbations(fp.mesh, delta_master)

    t_per_strut = np.exp(fp.mu_L_t + fp.sigma_L_t * g_t_mid)
    E_per_strut = np.exp(fp.mu_L_E + fp.sigma_L_E * g_E_mid)

    return forward_solve_heterogeneous(
        fp.mesh, full_delta, t_per_strut, E_per_strut,
        return_full=return_full)


if __name__ == "__main__":
    print("=" * 60)
    print("FieldProblem + forward_map smoke test")
    print("=" * 60)

    fp = build_field_problem()
    print(f"\nStochastic dimension: {fp.n_dim}")
    print()
    print(fp.kl_model.summary())

    xi0 = np.zeros(fp.n_dim)
    E_det, nu_det = forward_map(xi0, fp)
    print(f"\nDeterministic check (xi = 0):")
    print(f"  E*  = {E_det:.6e}")
    print(f"  nu* = {nu_det:.6f}")

    from honeycomb_fe import forward_solve
    E_ref, nu_ref = forward_solve(fp.spec.theta_mean_deg, fp.spec.l_mean,
                                  fp.spec.t_mean, fp.spec.E_mean)
    print(f"  scalar solver ref: E* = {E_ref:.6e}, nu* = {nu_ref:.6f}")

    print(f"  (small mismatch expected: xi=0 gives lognormal medians, not means)")

    rng = np.random.default_rng(2026)
    xi_rand = rng.standard_normal(fp.n_dim)
    t0 = time.time()
    E_s, nu_s = forward_map(xi_rand, fp)
    elapsed = (time.time() - t0) * 1000
    print(f"\nRandom sample (xi ~ N(0, I_{fp.n_dim})):")
    print(f"  E*  = {E_s:.6e}")
    print(f"  nu* = {nu_s:.6f}")
    print(f"  wall time: {elapsed:.1f} ms")

    print(f"\nRate benchmark: 100 samples")
    t0 = time.time()
    Es = np.zeros(100)
    nus = np.zeros(100)
    for i in range(100):
        xi_i = rng.standard_normal(fp.n_dim)
        Es[i], nus[i] = forward_map(xi_i, fp)
    elapsed = time.time() - t0
    print(f"  {100/elapsed:.0f} samples/s -> one sample = {elapsed*10:.2f} ms")
    print(f"  E* range over 100 samples: [{Es.min():.4e}, {Es.max():.4e}]")
    print(f"  nu* range over 100 samples: [{nus.min():.4f}, {nus.max():.4f}]")

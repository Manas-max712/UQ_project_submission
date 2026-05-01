from __future__ import annotations
import numpy as np
from typing import List

from honeycomb_fe import Mesh
from random_field import build_kl_field, KLField, KLModel


def master_node_positions(mesh: Mesh) -> np.ndarray:
    N = mesh.nodes.shape[0]
    slave_ids = set(mesh.pairs.keys())
    master_ids = [i for i in range(N) if i not in slave_ids]
    return mesh.nodes[master_ids].copy()


def master_node_ids(mesh: Mesh) -> List[int]:
    N = mesh.nodes.shape[0]
    slave_ids = set(mesh.pairs.keys())
    return sorted(i for i in range(N) if i not in slave_ids)


def strut_midpoints(mesh: Mesh) -> np.ndarray:
    p1 = mesh.nodes[mesh.elems[:, 0]]
    p2 = mesh.nodes[mesh.elems[:, 1]]
    return 0.5 * (p1 + p2)

def build_kl_model(mesh: Mesh,
                   ell_c: float = 1.5,
                   sigma_delta_ratio: float = 0.03,
                   l_nominal: float = 1.0,
                   energy_tol: float = 0.95) -> KLModel:
    master_pts = master_node_positions(mesh)
    midpts = strut_midpoints(mesh)

    sigma_delta = sigma_delta_ratio * l_nominal

    dx_field = build_kl_field(master_pts, ell_c=ell_c, sigma=sigma_delta,
                              mean=None, energy_tol=energy_tol)

    dy_field = build_kl_field(master_pts, ell_c=ell_c, sigma=sigma_delta,
                              mean=None, energy_tol=energy_tol)

    t_field = build_kl_field(midpts, ell_c=ell_c, sigma=1.0,
                             mean=None, energy_tol=energy_tol)

    E_field = build_kl_field(midpts, ell_c=ell_c, sigma=1.0,
                             mean=None, energy_tol=energy_tol)

    return KLModel(dx=dx_field, dy=dy_field, t=t_field, E=E_field)

def expand_node_perturbations(mesh: Mesh,
                              delta_master: np.ndarray) -> np.ndarray:
    N = mesh.nodes.shape[0]
    master_ids = master_node_ids(mesh)
    full_delta = np.zeros((N, 2))

    for k, i in enumerate(master_ids):
        full_delta[i] = delta_master[k]

    for s, (m, _shift) in mesh.pairs.items():
        full_delta[s] = full_delta[m]

    return full_delta

if __name__ == "__main__":
    from honeycomb_fe import build_honeycomb_rve

    mesh = build_honeycomb_rve(theta_deg=30.0, l=1.0, nx=2, ny=2)
    print(f"Mesh: {mesh.nodes.shape[0]} nodes, {mesh.elems.shape[0]} elements")
    print(f"       {len(mesh.pairs)} slave nodes, "
          f"{mesh.nodes.shape[0] - len(mesh.pairs)} master nodes")

    kl_model = build_kl_model(mesh, ell_c=1.5, sigma_delta_ratio=0.03)
    print()
    print(kl_model.summary())

    rng = np.random.default_rng(42)
    xi = rng.standard_normal(kl_model.n_dim)
    dx, dy, gt, gE = kl_model.reconstruct_all(xi)

    print(f"\nReconstructed fields at master nodes / midpoints:")
    print(f"  delta_x: shape {dx.shape}, range [{dx.min():+.4f}, {dx.max():+.4f}]")
    print(f"  delta_y: shape {dy.shape}, range [{dy.min():+.4f}, {dy.max():+.4f}]")
    print(f"  g_t:     shape {gt.shape}, range [{gt.min():+.3f}, {gt.max():+.3f}]")
    print(f"  g_E:     shape {gE.shape}, range [{gE.min():+.3f}, {gE.max():+.3f}]")

    delta_master = np.column_stack([dx, dy])
    full_delta = expand_node_perturbations(mesh, delta_master)
    print(f"\nExpanded delta: shape {full_delta.shape}")
    all_ok = True
    for s, (m, _shift) in mesh.pairs.items():
        if not np.allclose(full_delta[s], full_delta[m]):
            all_ok = False
            break
    print(f"  slave perturbations match masters: {all_ok}")

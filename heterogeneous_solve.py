from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from copy import deepcopy

from honeycomb_fe import (
    Mesh, frame_stiffness, build_periodic_transform,
    affine_displacement, solve_macro_strain,
    volume_average_stress, effective_properties,
    build_honeycomb_rve,
)


def perturb_mesh(mesh: Mesh, full_delta: np.ndarray) -> Mesh:
    assert full_delta.shape == mesh.nodes.shape, (
        f"full_delta {full_delta.shape} != mesh.nodes {mesh.nodes.shape}")
    new_nodes = mesh.nodes + full_delta
    return Mesh(nodes=new_nodes,
                elems=mesh.elems.copy(),
                Lx=mesh.Lx, Ly=mesh.Ly,
                pairs=deepcopy(mesh.pairs))

def assemble_heterogeneous(mesh: Mesh,
                           E_per_strut: np.ndarray,
                           t_per_strut: np.ndarray,
                           section: str = "rect") -> np.ndarray:
    N_nodes = mesh.nodes.shape[0]
    N_struts = mesh.elems.shape[0]
    assert E_per_strut.shape == (N_struts,), (
        f"E_per_strut shape {E_per_strut.shape} expected ({N_struts},)")
    assert t_per_strut.shape == (N_struts,), (
        f"t_per_strut shape {t_per_strut.shape} expected ({N_struts},)")

    K = np.zeros((3 * N_nodes, 3 * N_nodes))
    for k, (i, j) in enumerate(mesh.elems):
        t = t_per_strut[k]
        if section == "rect":
            A = t
            I = t**3 / 12.0
        elif section == "circle":
            A = np.pi * t**2 / 4.0
            I = np.pi * t**4 / 64.0
        else:
            raise ValueError(f"Unknown section {section!r}")
        E = E_per_strut[k]
        ke, _ = frame_stiffness(mesh.nodes[i], mesh.nodes[j], E, A, I)
        dofs = [3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2]
        for a in range(6):
            for b in range(6):
                K[dofs[a], dofs[b]] += ke[a, b]
    return K

def effective_stiffness_from_K(mesh: Mesh, K: np.ndarray) -> np.ndarray:
    T, shift_map, masters = build_periodic_transform(mesh)
    eps_modes = [
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 1.0]]),
        np.array([[0.0, 0.5], [0.5, 0.0]]),
    ]
    C = np.zeros((3, 3))
    for k_mode, eps in enumerate(eps_modes):
        u = solve_macro_strain(mesh, K, eps, T, shift_map, masters)
        sig = volume_average_stress(mesh, K, u)
        C[0, k_mode] = sig[0, 0]
        C[1, k_mode] = sig[1, 1]
        C[2, k_mode] = sig[0, 1]
    C = 0.5 * (C + C.T)
    return C

def forward_solve_heterogeneous(
    nominal_mesh: Mesh,
    full_delta: np.ndarray,
    t_per_strut: np.ndarray,
    E_per_strut: np.ndarray,
    section: str = "rect",
    return_full: bool = False,
):
    mesh_pert = perturb_mesh(nominal_mesh, full_delta)
    K = assemble_heterogeneous(mesh_pert, E_per_strut, t_per_strut, section=section)
    C = effective_stiffness_from_K(mesh_pert, K)
    E1, E2, nu12, nu21 = effective_properties(C)
    if return_full:
        return E1, nu12, C, mesh_pert
    return E1, nu12

if __name__ == "__main__":
    from honeycomb_fe import forward_solve

    theta_deg, l, t, E = 30.0, 1.0, 0.05, 200.0
    nx, ny = 2, 2

    E_ref, nu_ref = forward_solve(theta_deg, l, t, E, nx, ny)

    mesh = build_honeycomb_rve(theta_deg, l, nx, ny)
    N_nodes = mesh.nodes.shape[0]
    N_struts = mesh.elems.shape[0]
    full_delta = np.zeros((N_nodes, 2))
    t_arr = np.full(N_struts, t)
    E_arr = np.full(N_struts, E)

    E_het, nu_het = forward_solve_heterogeneous(
        mesh, full_delta, t_arr, E_arr)

    print("Regression test: heterogeneous solver with zero perturbations and")
    print("uniform (t, E) should EXACTLY reproduce the scalar solver.")
    print(f"  E_ref = {E_ref:.8e}   E_het = {E_het:.8e}   "
          f"rel diff = {abs(E_het - E_ref)/abs(E_ref):.2e}")
    print(f"  nu_ref = {nu_ref:.8e}   nu_het = {nu_het:.8e}   "
          f"rel diff = {abs(nu_het - nu_ref)/abs(nu_ref):.2e}")

    rng = np.random.default_rng(0)
    small_delta = 0.01 * rng.standard_normal((N_nodes, 2))
    for s, (m, _shift) in mesh.pairs.items():
        small_delta[s] = small_delta[m]
    t_arr_het = t * np.exp(0.05 * rng.standard_normal(N_struts))
    E_arr_het = E * np.exp(0.05 * rng.standard_normal(N_struts))

    E_pert, nu_pert = forward_solve_heterogeneous(
        mesh, small_delta, t_arr_het, E_arr_het)
    print()
    print("With small perturbations, the answer should move but stay finite:")
    print(f"  E* perturbed = {E_pert:.6e}  (ref {E_ref:.6e})")
    print(f"  nu* perturbed = {nu_pert:.6f}  (ref {nu_ref:.6f})")

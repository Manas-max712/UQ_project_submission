from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Dict


@dataclass
class Mesh:
    nodes: np.ndarray
    elems: np.ndarray
    Lx: float
    Ly: float
    pairs: Dict[int, Tuple[int, np.ndarray]] = field(default_factory=dict)


def _round_key(p: np.ndarray, tol: float = 1e-8) -> Tuple[int, int]:
    return (int(round(p[0] / tol)), int(round(p[1] / tol)))


def build_honeycomb_rve(theta_deg: float, l: float,
                        nx: int = 1, ny: int = 1) -> Mesh:
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    h = l
    ax = 2.0 * l * c
    ay = 2.0 * l * (1.0 + s)

    template = np.array([
        [0.0,   0.0],
        [l * c, l * s],
        [l * c, l * s + h],
        [0.0,   2.0 * l * s + h],
    ])

    bar_template = [
        (0, 1,  0,  0),
        (1, 0,  1,  0),
        (1, 2,  0,  0),
        (2, 3,  0,  0),
        (3, 2, -1,  0),
        (3, 0,  0,  1),
    ]

    Lx = nx * ax
    Ly = ny * ay

    nodes_list: List[np.ndarray] = []
    node_idx: Dict[Tuple[int, int], int] = {}

    def add_node(p: np.ndarray) -> int:
        k = _round_key(p)
        if k in node_idx:
            return node_idx[k]
        idx = len(nodes_list)
        node_idx[k] = idx
        nodes_list.append(p.copy())
        return idx

    interior_ids: Dict[Tuple[int, int, int], int] = {}
    for ix in range(nx):
        for iy in range(ny):
            for k_loc in range(4):
                p = template[k_loc] + np.array([ix * ax, iy * ay])
                interior_ids[(ix, iy, k_loc)] = add_node(p)

    pairs: Dict[int, Tuple[int, np.ndarray]] = {}

    def node_for(jx: int, jy: int, k_loc: int) -> int:
        if 0 <= jx < nx and 0 <= jy < ny:
            return interior_ids[(jx, jy, k_loc)]
        jx_in, jy_in = jx % nx, jy % ny
        shift = np.array([((jx - jx_in) // nx) * Lx,
                          ((jy - jy_in) // ny) * Ly])
        master = interior_ids[(jx_in, jy_in, k_loc)]
        p_ghost = nodes_list[master] + shift
        nid = add_node(p_ghost)
        pairs.setdefault(nid, (master, shift))
        return nid

    elems: List[Tuple[int, int]] = []
    for ix in range(nx):
        for iy in range(ny):
            for (li, lj, dcx, dcy) in bar_template:
                ni = node_for(ix, iy, li)
                nj = node_for(ix + dcx, iy + dcy, lj)
                elems.append((ni, nj))

    return Mesh(nodes=np.array(nodes_list),
                elems=np.array(elems, dtype=int),
                Lx=Lx, Ly=Ly, pairs=pairs)

def frame_stiffness(p1: np.ndarray, p2: np.ndarray,
                    E: float, A: float, I: float) -> Tuple[np.ndarray, float]:
    d = p2 - p1
    L = float(np.linalg.norm(d))
    cx, cy = d[0] / L, d[1] / L

    EA_L = E * A / L
    EI = E * I
    ke_local = np.zeros((6, 6))
    ke_local[0, 0] =  EA_L
    ke_local[0, 3] = -EA_L
    ke_local[3, 0] = -EA_L
    ke_local[3, 3] =  EA_L
    k1 = 12 * EI / L**3
    k2 =  6 * EI / L**2
    k3 =  4 * EI / L
    k4 =  2 * EI / L
    b = np.array([
        [ k1,  k2, -k1,  k2],
        [ k2,  k3, -k2,  k4],
        [-k1, -k2,  k1, -k2],
        [ k2,  k4, -k2,  k3],
    ])
    idx = [1, 2, 4, 5]
    for a, ia in enumerate(idx):
        for bb, ib in enumerate(idx):
            ke_local[ia, ib] = b[a, bb]

    R = np.zeros((6, 6))
    R[0, 0] =  cx; R[0, 1] = cy
    R[1, 0] = -cy; R[1, 1] = cx
    R[2, 2] = 1.0
    R[3, 3] =  cx; R[3, 4] = cy
    R[4, 3] = -cy; R[4, 4] = cx
    R[5, 5] = 1.0

    ke = R.T @ ke_local @ R
    return ke, L


def assemble(mesh: Mesh, E: float, A: float, I: float) -> np.ndarray:
    N = mesh.nodes.shape[0]
    K = np.zeros((3 * N, 3 * N))
    for (i, j) in mesh.elems:
        ke, _ = frame_stiffness(mesh.nodes[i], mesh.nodes[j], E, A, I)
        dofs = [3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2]
        for a in range(6):
            for bb in range(6):
                K[dofs[a], dofs[bb]] += ke[a, bb]
    return K

def build_periodic_transform(mesh: Mesh) -> Tuple[np.ndarray,
                                                  Dict[int, np.ndarray],
                                                  List[int]]:
    N = mesh.nodes.shape[0]
    slaves = mesh.pairs
    master_nodes = sorted(set(range(N)) - set(slaves.keys()))
    m_index = {n: k for k, n in enumerate(master_nodes)}
    M = len(master_nodes)

    T = np.zeros((3 * N, 3 * M))
    for n in master_nodes:
        k = m_index[n]
        T[3*n,     3*k]     = 1.0
        T[3*n + 1, 3*k + 1] = 1.0
        T[3*n + 2, 3*k + 2] = 1.0
    for s, (m, _shift) in slaves.items():
        k = m_index[m]
        T[3*s,     3*k]     = 1.0
        T[3*s + 1, 3*k + 1] = 1.0
        T[3*s + 2, 3*k + 2] = 1.0

    shift_map = {s: shift for s, (_m, shift) in slaves.items()}
    return T, shift_map, master_nodes


def affine_displacement(mesh: Mesh, eps_bar: np.ndarray,
                        shift_map: Dict[int, np.ndarray]) -> np.ndarray:
    N = mesh.nodes.shape[0]
    u0 = np.zeros(3 * N)
    for s, r in shift_map.items():
        du = eps_bar @ r
        u0[3*s]     = du[0]
        u0[3*s + 1] = du[1]
    return u0


def solve_macro_strain(mesh: Mesh, K: np.ndarray, eps_bar: np.ndarray,
                       T: np.ndarray, shift_map, master_nodes) -> np.ndarray:
    u0 = affine_displacement(mesh, eps_bar, shift_map)
    A = T.T @ K @ T
    b = -T.T @ K @ u0

    M = len(master_nodes)




    fixed = [0, 1]
    free = [i for i in range(3 * M) if i not in fixed]

    A_ff = A[np.ix_(free, free)]
    b_f = b[free]
    u_red = np.zeros(3 * M)
    u_red[free] = np.linalg.solve(A_ff, b_f)

    return T @ u_red + u0


def volume_average_stress(mesh: Mesh, K: np.ndarray,
                          u: np.ndarray) -> np.ndarray:
    V = mesh.Lx * mesh.Ly * 1.0
    Ku = K @ u
    sig = np.zeros((2, 2))
    for s, (_m, delta) in mesh.pairs.items():
        fx = Ku[3*s]
        fy = Ku[3*s + 1]
        sig[0, 0] += fx * delta[0]
        sig[0, 1] += fx * delta[1]
        sig[1, 0] += fy * delta[0]
        sig[1, 1] += fy * delta[1]
    sig /= V
    sig = 0.5 * (sig + sig.T)
    return sig


def effective_stiffness(mesh: Mesh, E: float, A: float, I: float) -> np.ndarray:
    K = assemble(mesh, E, A, I)
    T, shift_map, masters = build_periodic_transform(mesh)

    eps_modes = [
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 1.0]]),
        np.array([[0.0, 0.5], [0.5, 0.0]]),
    ]
    C = np.zeros((3, 3))
    for k, eps in enumerate(eps_modes):
        u = solve_macro_strain(mesh, K, eps, T, shift_map, masters)
        sig = volume_average_stress(mesh, K, u)
        C[0, k] = sig[0, 0]
        C[1, k] = sig[1, 1]
        C[2, k] = sig[0, 1]
    C = 0.5 * (C + C.T)
    return C


def effective_properties(C: np.ndarray) -> Tuple[float, float, float, float]:
    C11, C22, C12 = C[0, 0], C[1, 1], C[0, 1]
    E1 = C11 - C12**2 / C22
    E2 = C22 - C12**2 / C11
    nu12 = C12 / C22
    nu21 = C12 / C11
    return E1, E2, nu12, nu21

def forward_solve(theta_deg: float, l: float, t: float, E: float,
                  nx: int = 2, ny: int = 2,
                  return_full: bool = False):
    A = t
    I = t**3 / 12.0
    mesh = build_honeycomb_rve(theta_deg, l, nx, ny)
    C = effective_stiffness(mesh, E, A, I)
    E1, E2, nu12, nu21 = effective_properties(C)
    if return_full:
        return E1, nu12, C, mesh
    return E1, nu12

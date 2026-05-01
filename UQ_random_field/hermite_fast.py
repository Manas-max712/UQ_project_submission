from __future__ import annotations
import numpy as np
from typing import List, Tuple
import time
import math


def gen_multiindices(d: int, p: int) -> np.ndarray:
    result = []

    def recurse(prefix: List[int], remaining_dims: int, remaining_budget: int):
        if remaining_dims == 0:
            result.append(tuple(prefix))
            return
        for k in range(remaining_budget + 1):
            recurse(prefix + [k], remaining_dims - 1, remaining_budget - k)

    recurse([], d, p)
    arr = np.array(result, dtype=int)
    order = np.lexsort([arr[:, i] for i in range(d - 1, -1, -1)] + [arr.sum(axis=1)])
    return arr[order]


def hermite_tableau(xi: np.ndarray, p: int) -> np.ndarray:
    d, M = xi.shape
    H = np.zeros((p + 1, d, M))
    H[0] = 1.0
    if p >= 1:
        H[1] = xi
    for k in range(1, p):
        H[k + 1] = xi * H[k] - k * H[k - 1]
    return H

def compute_norms(multi_indices: np.ndarray) -> np.ndarray:
    logfac = np.array([math.lgamma(k + 1) for k in range(multi_indices.max() + 1)])
    log_norms = 0.5 * logfac[multi_indices].sum(axis=1)
    return np.exp(log_norms)


def eval_basis_chunk(hermite_tab_chunk: np.ndarray,
                     multi_indices: np.ndarray,
                     norms: np.ndarray) -> np.ndarray:
    p_plus_1, d, M = hermite_tab_chunk.shape
    N_basis = multi_indices.shape[0]

    Phi = np.ones((N_basis, M))
    for i in range(d):
        Phi *= hermite_tab_chunk[multi_indices[:, i], i, :]
    Phi /= norms[:, None]
    return Phi

def project_fast(xi_grid: np.ndarray, weights: np.ndarray,
                 f_values: np.ndarray,
                 multi_indices: np.ndarray, norms: np.ndarray,
                 p: int,
                 batch_size: int = 4000,
                 verbose: bool = True) -> np.ndarray:
    d, M = xi_grid.shape
    N_basis = multi_indices.shape[0]
    coeffs = np.zeros(N_basis)
    wf = weights * f_values

    t0 = time.time()
    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        xi_chunk = xi_grid[:, start:end]
        H_chunk = hermite_tableau(xi_chunk, p)
        Phi = eval_basis_chunk(H_chunk, multi_indices, norms)
        coeffs += Phi @ wf[start:end]

        if verbose and M >= 5000 and ((start // batch_size) % max(1, (M // batch_size) // 10) == 0):
            elapsed = time.time() - t0
            frac = end / M
            rate = end / elapsed if elapsed > 0 else 0
            eta = (M - end) / rate if rate > 0 else 0
            print(f"    projection {end:>6d}/{M:<6d} ({frac*100:5.1f}%) "
                  f"rate = {rate:6.0f} pts/s  eta = {eta:5.1f}s")

    elapsed = time.time() - t0
    if verbose:
        print(f"  fast projection done in {elapsed:.2f} s")
    return coeffs


def eval_surrogate_fast(xi_samples: np.ndarray,
                        coeffs: np.ndarray,
                        multi_indices: np.ndarray, norms: np.ndarray,
                        p: int, batch_size: int = 5000) -> np.ndarray:
    d, N_samples = xi_samples.shape
    out = np.zeros(N_samples)
    for start in range(0, N_samples, batch_size):
        end = min(start + batch_size, N_samples)
        xi_chunk = xi_samples[:, start:end]
        H_chunk = hermite_tableau(xi_chunk, p)
        Phi = eval_basis_chunk(H_chunk, multi_indices, norms)
        out[start:end] = coeffs @ Phi
    return out

if __name__ == "__main__":
    for d, p in [(2, 3), (4, 2), (4, 4), (30, 2), (30, 3)]:
        alphas = gen_multiindices(d, p)
        expected = math.comb(p + d, d)
        assert alphas.shape == (expected, d),\
            f"d={d}, p={p}: got {alphas.shape}, expected ({expected}, {d})"
        assert (alphas.sum(axis=1) <= p).all()
    print("Multi-index enumeration: OK")

    rng = np.random.default_rng(0)
    N = 200000
    xi = rng.standard_normal((2, N))
    H = hermite_tableau(xi, 4)
    for k in range(5):
        m = H[k, 0, :].mean()
        expected = 1.0 if k == 0 else 0.0
        assert abs(m - expected) < 0.02, f"H_{k} mean off: {m}"
    m2 = (H[2, 0, :] ** 2).mean()
    assert abs(m2 - 2.0) < 0.1, f"E[H_2^2] = {m2}, expected 2"
    m3 = (H[3, 0, :] ** 2).mean()
    assert abs(m3 - 6.0) < 0.3, f"E[H_3^2] = {m3}, expected 6"
    print("Hermite tableau + orthogonality: OK")

    d, p = 4, 3
    alphas = gen_multiindices(d, p)
    norms = compute_norms(alphas)
    xi = rng.standard_normal((d, 500000))
    H = hermite_tableau(xi, p)
    Phi = eval_basis_chunk(H, alphas, norms)
    mean_vec = Phi.mean(axis=1)
    norm_vec = (Phi ** 2).mean(axis=1)
    zero_idx = 0
    assert abs(mean_vec[zero_idx] - 1.0) < 0.01
    assert (np.abs(mean_vec[1:]) < 0.05).all()
    assert (np.abs(norm_vec - 1.0) < 0.05).all()
    print(f"Orthonormality (d={d}, p={p}, N_basis={len(alphas)}): OK")

    d, p = 30, 3
    t0 = time.time()
    alphas = gen_multiindices(d, p)
    norms = compute_norms(alphas)
    print(f"Setup d={d}, p={p}: {time.time()-t0:.3f}s, N_basis={len(alphas)}")

    xi_test = rng.standard_normal((d, 1000))
    t0 = time.time()
    H_tab = hermite_tableau(xi_test, p)
    Phi = eval_basis_chunk(H_tab, alphas, norms)
    print(f"Evaluate basis at 1000 pts: {time.time()-t0:.2f}s, shape {Phi.shape}")

    n_full = 40000
    t_est = (time.time() - t0) * n_full / 1000
    print(f"Extrapolated time for {n_full} pts: ~{t_est:.1f}s")

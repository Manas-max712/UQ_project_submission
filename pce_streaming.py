from __future__ import annotations
import numpy as np
import chaospy as cp
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")


def project_streaming(expansion, xi_grid: np.ndarray, weights: np.ndarray,
                      f_values: np.ndarray,
                      batch_size: int = 1000,
                      verbose: bool = True) -> np.ndarray:
    n_dim, M = xi_grid.shape
    n_basis = len(expansion)
    coeffs = np.zeros(n_basis)

    wf = weights * f_values

    import time
    t0 = time.time()
    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        xi_chunk = xi_grid[:, start:end]
        Phi_chunk = np.asarray(expansion(*xi_chunk))
        coeffs += Phi_chunk @ wf[start:end]

        if verbose and M >= 5000 and ((start // batch_size) % max(1, (M // batch_size) // 10) == 0):
            elapsed = time.time() - t0
            frac = end / M
            rate = end / elapsed if elapsed > 0 else 0
            eta = (M - end) / rate if rate > 0 else 0
            print(f"    projection {end:>6d}/{M:<6d} ({frac*100:5.1f}%) "
                  f"rate = {rate:5.0f} pts/s  eta = {eta:5.1f}s")

    elapsed = time.time() - t0
    if verbose:
        print(f"  projection done in {elapsed:.1f} s")
    return coeffs


def eval_surrogate_streaming(expansion, coeffs: np.ndarray,
                             xi_samples: np.ndarray,
                             batch_size: int = 2000) -> np.ndarray:
    n_dim, N_samples = xi_samples.shape
    f_hat = np.zeros(N_samples)
    for start in range(0, N_samples, batch_size):
        end = min(start + batch_size, N_samples)
        xi_chunk = xi_samples[:, start:end]
        Phi_chunk = np.asarray(expansion(*xi_chunk))
        f_hat[start:end] = coeffs @ Phi_chunk
    return f_hat

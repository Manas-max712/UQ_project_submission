from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional


def squared_exponential(X: np.ndarray, ell_c: float,
                        sigma: float = 1.0) -> np.ndarray:
    diff = X[:, None, :] - X[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    C = sigma * sigma * np.exp(-0.5 * dist2 / (ell_c * ell_c))
    return C

@dataclass
class KLField:
    points: np.ndarray
    mean: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray
    total_var: float
    energy_frac: float
    n_dim: int

    def reconstruct(self, xi: np.ndarray) -> np.ndarray:
        if xi.shape != (self.n_dim,):
            raise ValueError(f"xi has shape {xi.shape}, expected ({self.n_dim},)")
        scale = np.sqrt(self.eigvals) * xi
        return self.mean + self.eigvecs @ scale

    def sample(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        xi = rng.standard_normal(self.n_dim)
        return xi, self.reconstruct(xi)


def build_kl_field(points: np.ndarray,
                   ell_c: float,
                   sigma: float = 1.0,
                   mean: Optional[np.ndarray] = None,
                   energy_tol: float = 0.95,
                   max_modes: Optional[int] = None) -> KLField:
    N = points.shape[0]
    if mean is None:
        mean = np.zeros(N)

    C = squared_exponential(points, ell_c, sigma)

    eigvals_all, eigvecs_all = np.linalg.eigh(C)

    eigvals_all = eigvals_all[::-1]
    eigvecs_all = eigvecs_all[:, ::-1]

    eigvals_all = np.clip(eigvals_all, 0.0, None)

    total_var = eigvals_all.sum()
    cumulative = np.cumsum(eigvals_all) / total_var

    K = int(np.searchsorted(cumulative, energy_tol)) + 1
    if max_modes is not None:
        K = min(K, max_modes)
    K = max(K, 1)

    eigvals = eigvals_all[:K]
    eigvecs = eigvecs_all[:, :K]
    energy_frac = float(cumulative[K - 1])

    return KLField(points=points, mean=mean,
                   eigvals=eigvals, eigvecs=eigvecs,
                   total_var=float(total_var),
                   energy_frac=energy_frac,
                   n_dim=K)

@dataclass
class KLModel:
    dx: KLField
    dy: KLField
    t:  KLField
    E:  KLField

    @property
    def n_dim(self) -> int:
        return self.dx.n_dim + self.dy.n_dim + self.t.n_dim + self.E.n_dim

    def split(self, xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray]:
        K1, K2, K3 = self.dx.n_dim, self.dy.n_dim, self.t.n_dim
        xi_dx = xi[:K1]
        xi_dy = xi[K1:K1 + K2]
        xi_t  = xi[K1 + K2:K1 + K2 + K3]
        xi_E  = xi[K1 + K2 + K3:]
        return xi_dx, xi_dy, xi_t, xi_E

    def reconstruct_all(self, xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray]:
        xi_dx, xi_dy, xi_t, xi_E = self.split(xi)
        return (self.dx.reconstruct(xi_dx),
                self.dy.reconstruct(xi_dy),
                self.t.reconstruct(xi_t),
                self.E.reconstruct(xi_E))

    def summary(self) -> str:
        lines = [
            f"KLModel summary:",
            f"  delta_x field: {self.dx.n_dim} modes, "
                f"{self.dx.energy_frac*100:.1f}% variance captured, "
                f"N points = {self.dx.points.shape[0]}",
            f"  delta_y field: {self.dy.n_dim} modes, "
                f"{self.dy.energy_frac*100:.1f}% variance captured, "
                f"N points = {self.dy.points.shape[0]}",
            f"  t field:       {self.t.n_dim} modes, "
                f"{self.t.energy_frac*100:.1f}% variance captured, "
                f"N points = {self.t.points.shape[0]}",
            f"  E field:       {self.E.n_dim} modes, "
                f"{self.E.energy_frac*100:.1f}% variance captured, "
                f"N points = {self.E.points.shape[0]}",
            f"  total stochastic dimension: {self.n_dim}",
        ]
        return "\n".join(lines)

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    xs = np.linspace(0, 3, 5)
    ys = np.linspace(0, 3, 4)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    kl = build_kl_field(points, ell_c=1.5, sigma=1.0, energy_tol=0.95)
    print(f"Built K-L field on {points.shape[0]} points.")
    print(f"  eigenvalues (top 10): {kl.eigvals[:10]}")
    print(f"  modes retained: {kl.n_dim}")
    print(f"  energy captured: {kl.energy_frac*100:.2f}%")

    xi, f_vals = kl.sample(rng)
    print(f"  sample xi = {xi}")
    print(f"  field range: [{f_vals.min():.3f}, {f_vals.max():.3f}]")

    M = 20000
    samples = np.zeros((M, points.shape[0]))
    for i in range(M):
        _, samples[i] = kl.sample(rng)
    empirical_var = samples.var(axis=0).mean()
    print(f"  empirical marginal variance: {empirical_var:.4f} (target 1.0)")

from typing_extensions import override
import numpy as np
import cupy as cp
from numpy.random import random

from dgsp.estimators.monte_carlo.base import MonteCarloFilter


def systematic_resample(weights, backend):
    N = len(weights)

    positions = (random() + backend.arange(N)) / N

    indexes = backend.zeros(N, "i")
    cumulative_sum = backend.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


class ParticleFilter(MonteCarloFilter):
    n_particles: int
    particles: np.ndarray
    weights: np.ndarray

    def __init__(
        self, n_particles: int = 1000, backend_type: str = "numpy", bootstrap=False
    ) -> None:
        super().__init__(n_particles, backend_type)
        self.n_eff = self.n_particles // 2

    def resample(self) -> None:
        backend = np if self.backend_type == "numpy" else cp
        indices = systematic_resample(self.weights, backend)
        self.n_particles = len(indices)
        self.particles[:] = self.particles[indices]
        self.weights = backend.resize(self.weights, self.n_particles)
        self.weights.fill(1.0 / self.n_particles)

    @override
    def update(self, data: np.ndarray) -> None:
        backend = np if self.backend_type == "numpy" else cp

        observations_est = self.observation(self.particles, batched=True)

        def normal_pdf(mean, cov, data):
            k = mean.shape[0]
            data_c = data - mean

            def batched_quad_form(X, Sigma):
                Y = backend.linalg.solve(Sigma, X.T).T
                q = backend.einsum("bi,bi->b", X, Y)
                return q

            quad = batched_quad_form(data_c, cov)
            log_pdf = (
                k * backend.log(2 * backend.pi)
                + backend.log(backend.linalg.det(cov))
                + quad
            )
            return backend.exp(-0.5 * log_pdf)

        likelihood = normal_pdf(data, self.R, observations_est)
        self.weights *= likelihood
        self.weights += 1e-300
        self.weights /= backend.sum(self.weights)

        n_eff = 1 / backend.sum(self.weights**2)

        if n_eff < self.n_eff:
            self.resample()

        return super().update(data)

from typing_extensions import override
import numpy as np
from scipy.stats import multivariate_normal as norm
from filterpy.monte_carlo import systematic_resample

from dgsp.estimators.monte_carlo.base import MonteCarloFilter
from dgsp.functions import (
    observation,
)


class ParticleFilter(MonteCarloFilter):
    n_particles: int
    particles: np.ndarray
    weights: np.ndarray

    def __init__(self, n_particles: int = 1000, bootstrap=False) -> None:
        super().__init__(n_particles)
        self.n_eff = self.n_particles // 2

    def resample(self) -> None:
        indices = systematic_resample(self.weights)
        self.n_particles = len(indices)
        self.particles[:] = self.particles[indices]
        self.weights.resize(self.n_particles)
        self.weights.fill(1.0 / self.n_particles)

    @override
    def update(self, data: np.ndarray) -> None:
        observations_est = np.apply_along_axis(
            lambda x: observation(x, self.time), arr=self.particles, axis=1
        )

        likelihood = norm(mean=data, cov=self.R).pdf(observations_est)
        self.weights *= likelihood
        self.weights += 1e-300
        self.weights /= np.sum(self.weights)

        n_eff = 1 / np.sum(self.weights**2)

        if n_eff < self.n_eff:
            self.resample()

        return super().update(data)

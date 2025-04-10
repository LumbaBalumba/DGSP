from typing_extensions import override
import numpy as np
from scipy.stats import multivariate_normal as norm
from filterpy.monte_carlo import systematic_resample

from dgsp.estimators.base import Estimator
from dgsp.functions import (
    initial,
    transition,
    observation,
    transition_noise,
    dim_state,
)


class ParticleFilter(Estimator):
    n_particles: int
    particles: np.ndarray
    weights: np.ndarray

    def __init__(self, dt: float, n_particles: int = 1000, bootstrap=False) -> None:
        super().__init__(dt)
        self.n_particles = n_particles
        self.particles = np.random.multivariate_normal(
            mean=initial, cov=0.3 * np.eye(dim_state), size=self.n_particles
        )
        self.weights = np.ones(n_particles) / n_particles
        self.n_eff = self.n_particles // 2

    @override
    def predict(self) -> None:
        self.particles += self.dt * np.apply_along_axis(
            lambda x: transition(x, self.time), arr=self.particles, axis=1
        )
        for i in range(self.n_particles):
            self.particles[i] += transition_noise()

        state_est = np.average(self.particles, weights=self.weights, axis=0)
        k_est = np.diag(
            np.average((state_est - self.particles) ** 2, weights=self.weights, axis=0)
        )

        self.state.append(state_est)
        self.k.append(k_est)
        return super().predict()

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

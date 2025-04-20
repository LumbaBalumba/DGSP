from typing_extensions import override
import numpy as np

from dgsp.estimators.base import Estimator
from dgsp.functions import transition


class MonteCarloFilter(Estimator):
    n_particles: int
    particles: np.ndarray

    def __init__(self, n_particles: int = 1000) -> None:
        super().__init__()
        self.n_particles = n_particles
        self.particles = np.random.multivariate_normal(
            mean=self.state[0], cov=self.k[0], size=self.n_particles
        )
        self.weights = np.ones(self.n_particles) / n_particles

    @override
    def predict(self) -> None:
        self.particles += self.dt * np.apply_along_axis(
            lambda x: transition(x, self.time) + self.transition_noise(),
            arr=self.particles,
            axis=1,
        )

        state_est = np.average(self.particles, weights=self.weights, axis=0)
        k_est = np.cov(self.particles, rowvar=False, ddof=1, aweights=self.weights)

        self.state.append(state_est)
        self.k.append(k_est)
        return super().predict()

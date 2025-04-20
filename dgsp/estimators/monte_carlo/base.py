from typing_extensions import override
import numpy as np

from dgsp.estimators.base import Estimator


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
        self.particles = np.apply_along_axis(
            lambda x: self.transition(x) + self.transition_noise(),
            arr=self.particles,
            axis=1,
        )

        state_est = np.average(self.particles, weights=self.weights, axis=0)
        k_est = np.diag(
            np.average((state_est - self.particles) ** 2, weights=self.weights, axis=0)
        )

        self.state.append(state_est)
        self.k.append(k_est)
        super().predict()

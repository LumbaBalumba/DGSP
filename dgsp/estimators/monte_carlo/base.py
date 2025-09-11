from typing_extensions import override
import numpy as np
import cupy as cp

from dgsp.estimators.base import Estimator


class MonteCarloFilter(Estimator):
    n_particles: int
    particles: np.ndarray

    def __init__(self, n_particles: int = 1000, backend_type: str = "numpy") -> None:
        super().__init__(backend_type)

        backend = np if self.backend_type == "numpy" else cp

        self.n_particles = n_particles
        self.particles = backend.random.multivariate_normal(
            mean=self.state[0], cov=self.k[0], size=self.n_particles
        )
        self.weights = backend.ones(self.n_particles) / n_particles

    @override
    def predict(self) -> None:
        backend = np if self.backend_type == "numpy" else cp

        self.particles = backend.apply_along_axis(
            lambda x: self.transition(x),
            arr=self.particles,
            axis=1,
        )
        self.particles += self.transition_noise(self.n_particles)

        state_est = backend.average(self.particles, weights=self.weights, axis=0)
        k_est = backend.diag(
            backend.average(
                (state_est - self.particles) ** 2, weights=self.weights, axis=0
            )
        )

        self.state.append(state_est)
        self.k.append(k_est)
        super().predict()

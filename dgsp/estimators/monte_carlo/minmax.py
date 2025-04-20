import numpy as np

from dgsp.functions import (
    observation_noise,
    transition,
    observation,
    initial_guess,
    dim_state,
    dim_observation,
    transition_noise,
)
from dgsp.estimators.base import Estimator


class MinMaxFilter(Estimator):

    def __init__(self, dt: float, n_particles: int = 1000) -> None:
        super().__init__(dt)

        self.n_particles = n_particles
        self.particles = np.random.multivariate_normal(
            mean=initial_guess, cov=self.P, size=self.n_particles
        )

        self._m_x = initial_guess.copy()
        self._m_z = observation(initial_guess, self.time)
        self._R_xy = np.zeros((dim_state, dim_observation))
        self._R_yy = np.eye(dim_observation)

    def predict(self) -> None:
        for i in range(self.n_particles):
            self.particles[i] += transition(self.particles[i], self.time)
            self.particles[i] += transition_noise()

        obs_syn = np.array(
            [
                observation(particle, self.time) + observation_noise()
                for particle in self.particles
            ]
        )

        self._m_x = np.mean(self.particles, axis=0)
        self._m_z = np.mean(obs_syn, axis=0)

        Xc = self.particles - self._m_x
        Zc = obs_syn - self._m_z

        self._R_xy = (Xc.T @ Zc) / (self.n_particles - 1)
        self._R_yy = (Zc.T @ Zc) / (self.n_particles - 1)

        eps = 1e-9 * np.eye(dim_observation)
        self._R_yy += eps

        R_xx = (Xc.T @ Xc) / (self.n_particles - 1)
        self.state.append(self._m_x.copy())
        self.k.append(R_xx.copy())

        return super().predict()

    def update(self, data: np.ndarray) -> None:
        K = self._R_xy @ np.linalg.inv(self._R_yy)

        innov = data - self._m_z
        self.x = self._m_x + K @ innov

        R_xx = np.cov(self.particles.T, bias=False)
        self.P = R_xx - K @ self._R_yy @ K.T

        self.state[-1] = self.x.copy()
        self.k[-1] = self.P.copy()

        return super().update(data)

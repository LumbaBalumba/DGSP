import numpy as np
from numpy.linalg import inv

from dgsp.functions import (
    transition,
    transition_j,
    observation,
    observation_j,
)
from dgsp.estimators.base import Estimator


class ExtendedKalmanFilter(Estimator):

    def __init__(self, dt: float, order: int = 1, square_root: bool = False) -> None:
        super().__init__(dt)
        self.x = self.state[-1].copy()

    def predict(self) -> None:
        F = transition_j(self.x, self.time)
        self.x = self.x + transition(self.x, self.time) * self.dt

        self.P = F @ self.P @ F.T + self.Q
        self.state.append(self.x.copy())
        self.k.append(self.P.copy())
        super().predict()

    def update(self, data: np.ndarray) -> None:
        H = observation_j(self.x, self.time)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ inv(S)

        y = data - observation(self.x, self.time)
        self.x += K @ y
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P

        self.state[-1] = self.x.copy()
        self.k[-1] = self.P.copy()

        super().update(data)

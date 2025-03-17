from typing_extensions import override

import numpy as np

from dgsp.estimators import Estimator
from dgsp.functions import Q, transition


class UnscentedKalmanFilter(Estimator):
    n_points: int

    def __init__(
        self,
        dt: float,
        state: np.ndarray,
        k: np.ndarray,
    ) -> None:
        super().__init__(dt, state, k)
        self.n_points = 4

    def sigma_points(self) -> tuple[np.ndarray, np.ndarray]:
        w = np.zeros(2 * self.n_points + 1)
        points = np.zeros((2 * self.n_points + 1, 4))

        w[0] = 0.5
        w[1:] = (1 - w[0]) / (2 * self.n_points)

        points[0] = self.state

        for i in range(1, self.n_points + 1):
            points[i] = (
                points[0]
                + np.linalg.cholesky(self.k * self.n_points / (1 - w[0]))[:, i - 1]
            )

        for i in range(self.n_points + 1, 2 * self.n_points + 1):
            points[i] = (
                points[0]
                - np.linalg.cholesky(self.k * self.n_points / (1 - w[0]))[
                    :, i - self.n_points - 1
                ]
            )

        return points, w

    @override
    def predict_step(self) -> None:
        points, w = self.sigma_points()
        state_est = np.average(
            [transition(*point) for point in points], axis=0, weights=w
        )
        k_est = np.average([(transition(point) - state_est) @ (transition(point) - state_est).T] for point in points, axis=0, weights=w) + Q.diagonal() @ Q.diagonal().T
        return super().predict_step()

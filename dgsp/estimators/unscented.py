from numba import numba
from typing_extensions import override

import numpy as np

from dgsp.estimators.base import Estimator
from dgsp.functions import Q, R, transition, measurement


@numba.njit
def square(x: np.ndarray) -> np.ndarray:
    return x.reshape((-1, 1)) @ x.reshape((-1, 1)).T


@numba.njit
def fix(x: np.ndarray) -> np.ndarray:
    if not np.all(np.linalg.eigvals(x) > 0):
        x += np.eye(len(x)) * 1e-7
    return x


class UnscentedKalmanFilter(Estimator):
    n_points: int

    def __init__(
        self,
        dt: float,
    ) -> None:
        super().__init__(dt, np.zeros(4), Q)
        self.n_points = 4

    @staticmethod
    def sigma_points(
        n_points: int, state: np.ndarray, k: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        w = np.zeros(2 * n_points + 1)
        points = np.zeros((2 * n_points + 1, 4))

        w[0] = 0.5
        w[1:] = (1 - w[0]) / (2 * n_points)

        points[0] = state

        A = k * n_points / (1 - w[0])

        L = np.linalg.cholesky(A)

        for i in range(1, n_points + 1):
            points[i] = points[0] + L[:, i - 1]

        for i in range(n_points + 1, 2 * n_points + 1):
            points[i] = points[0] - L[:, i - n_points - 1]

        return points, w

    @override
    def predict_step(self) -> None:
        self.k[-1] = fix(self.k[-1])

        state = self.state[-1]
        k = self.k[-1]

        points, w = self.sigma_points(self.n_points, state, k)
        state_est = np.average(
            [transition(point) * self.dt + state for point in points], axis=0, weights=w
        ).reshape((-1))
        k_est = np.average(
            [
                square(transition(point) * self.dt + state - state_est)
                for point in points
            ],
            axis=0,
            weights=w,
        ).reshape((4, 4)) + square(np.array(Q.diagonal()))
        self.state.append(state_est)
        self.k.append(k_est)
        return super().predict_step()

    @override
    def correct_step(self, data: np.ndarray) -> None:
        self.k[-1] = fix(self.k[-1])
        points, w = self.sigma_points(self.n_points, self.state[-1], self.k[-1])
        measurement_est = np.average(
            [measurement(point) for point in points], axis=0, weights=w
        ).reshape((-1))

        kappa = np.average(
            [square(measurement(point) - measurement_est) for point in points],
            axis=0,
            weights=w,
        ).reshape((2, 2)) + square(np.array(R.diagonal()))

        mu = np.average(
            [
                (point - self.state[-1]).reshape((-1, 1))
                @ (measurement(point) - measurement_est).reshape((-1, 1)).T
                for point in points
            ],
            axis=0,
            weights=w,
        ).reshape((4, 2))

        kappa_inv = np.linalg.inv(kappa)

        self.state[-1] += mu @ kappa_inv @ (data - measurement_est)
        self.k[-1] -= mu @ kappa_inv @ mu.T

        return super().correct_step(data)

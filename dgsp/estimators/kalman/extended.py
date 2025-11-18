from typing_extensions import override
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter as EKF

from dgsp.functions import (
    dim_state,
    dim_observation,
)
from dgsp.estimators.base import Estimator


class ExtendedKalmanFilter(Estimator):
    kf: EKF

    def __init__(self, order: int = 1, square_root: bool = False) -> None:
        super().__init__()
        self.kf = EKF(dim_x=dim_state, dim_z=dim_observation)
        self.kf.x = self.state[0]
        self.kf.P = self.P
        self.kf.Q = self.Q
        self.kf.R = self.R

    @override
    def predict(self) -> None:
        self.kf.F = self.transition_j(self.kf.x) * self.dt + np.eye(dim_state)
        self.kf.predict()

        self.state.append(self.kf.x)
        self.k.append(self.kf.P)

        return super().predict()

    @override
    def update(self, data: np.ndarray) -> None:
        self.kf.update(data, self.observation_j, self.observation)

        self.state[-1] = self.kf.x
        self.k[-1] = self.kf.P

        super().update(data)

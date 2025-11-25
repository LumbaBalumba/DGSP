from typing_extensions import override
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter as EKF

from dgsp.estimators.kalman.base import BaseKalmanFilter
from dgsp.functions import (
    dim_state,
    dim_observation,
)


class ExtendedKalmanFilter(BaseKalmanFilter):
    def __init__(self, order: int = 1, square_root: bool = False) -> None:
        super().__init__(EKF, lambda: dict(dim_x=dim_state, dim_z=dim_observation))

    @override
    def predict(self, *args, **kwargs) -> None:
        self.kf.F = self.transition_j(self.kf.x) * self.dt + np.eye(dim_state)
        return super().predict(*args, **kwargs)

    @override
    def update(self, data: np.ndarray, *args, **kwargs) -> None:
        super().update(
            data,
            self.observation_j,
            self.observation,
            *args,
            **kwargs,
        )

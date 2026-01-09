from typing_extensions import override
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter as EKF

from dgsp.estimators.kalman.base import BaseKalmanFilter
from dgsp.functions import (
    dim_state,
    dim_observation,
)


class ExtendedKalmanFilter(BaseKalmanFilter):
    order: int

    def __init__(self, order: int = 1, square_root: bool = False) -> None:
        super().__init__(EKF, lambda: dict(dim_x=dim_state, dim_z=dim_observation))
        self.order = order

    @override
    def predict(self, *args, **kwargs) -> None:
        F = self.transition_j(self.kf.x) * self.dt + np.eye(dim_state)
        if self.order == 2:
            F += 0.5 * self.kf.x.T @ self.transition_h(self.kf.x) * self.dt**2
        self.kf.F = F
        return super().predict(*args, **kwargs)

    @override
    def update(self, data: np.ndarray, *args, **kwargs) -> None:
        match self.order:
            case 1:
                f = self.observation_j
            case 2:
                f = lambda x: self.observation_j(x) + 0.5 * x.T @ self.observation_h(x)
            case _:
                f = None
        super().update(
            data,
            f,
            self.observation,
            *args,
            **kwargs,
        )

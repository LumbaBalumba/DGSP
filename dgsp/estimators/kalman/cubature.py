from typing_extensions import override

import numpy as np
from filterpy_fixed.ckf import CubatureKalmanFilter as CKF

from dgsp.estimators.base import Estimator
from dgsp.functions import (
    dim_state,
    dim_observation,
)


class CubatureKalmanFilter(Estimator):
    kf: CKF

    def __init__(self, square_root: bool = False) -> None:
        super().__init__()

        self.kf = CKF(
            dim_x=dim_state,
            dim_z=dim_observation,
            dt=self.dt,
            fx=lambda x, dt: self.transition(x, dt),
            hx=lambda x: self.observation(x),
        )
        self.kf.x = self.state[0]
        self.kf.P = self.P
        self.kf.Q = self.Q
        self.kf.R = self.R

    @override
    def predict(self) -> None:
        self.kf.predict()

        self.state.append(self.kf.x.copy())
        self.k.append(self.kf.P.copy())

        return super().predict()

    @override
    def update(self, data: np.ndarray) -> None:
        self.kf.update(data)

        self.state[-1] = self.kf.x.copy()
        self.k[-1] = self.kf.P.copy()

        return super().update(data)

from typing_extensions import override

import numpy as np
from filterpy_fixed.ckf import CubatureKalmanFilter as CKF

from dgsp.estimators.base import Estimator
from dgsp.functions import (
    transition,
    observation,
    initial,
    dim_state,
    dim_observation,
)


class CubatureKalmanFilter(Estimator):
    kf: CKF

    def __init__(self, dt: float, square_root: bool = False) -> None:
        super().__init__(dt)

        def fx(x: np.ndarray, dt: float) -> np.ndarray:
            return transition(x, self.time) * dt + x

        def hx(x: np.ndarray) -> np.ndarray:
            return observation(x, self.time)

        self.kf = CKF(
            dim_x=dim_state,
            dim_z=dim_observation,
            dt=self.dt,
            fx=fx,
            hx=hx,
        )

        self.kf.x = initial

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

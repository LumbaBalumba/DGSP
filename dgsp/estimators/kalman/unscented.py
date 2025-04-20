from typing_extensions import override

import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF, MerweScaledSigmaPoints

from dgsp.estimators.base import Estimator
from dgsp.functions import (
    transition,
    observation,
    dim_state,
    dim_observation,
)


class UnscentedKalmanFilter(Estimator):
    kf: UKF

    def __init__(self, square_root: bool = False) -> None:
        super().__init__()

        def fx(x: np.ndarray, dt: float) -> np.ndarray:
            return transition(x, self.time) * dt + x

        def hx(x: np.ndarray) -> np.ndarray:
            return observation(x, self.time)

        points = MerweScaledSigmaPoints(dim_state, alpha=0.1, beta=2.0, kappa=-1.0)

        self.kf = UKF(
            dt=self.dt,
            dim_x=dim_state,
            dim_z=dim_observation,
            fx=fx,
            hx=hx,
            points=points,
        )
        self.kf.x = self.state[0]
        self.kf.P = self.P
        self.kf.Q = self.Q
        self.kf.R = self.R

    @override
    def predict(self) -> None:
        self.kf.predict()

        self.state.append(self.kf.x)
        self.k.append(self.kf.P)

        return super().predict()

    @override
    def update(self, data: np.ndarray) -> None:
        self.kf.update(data)

        self.state[-1] = self.kf.x
        self.k[-1] = self.kf.P

        return super().update(data)

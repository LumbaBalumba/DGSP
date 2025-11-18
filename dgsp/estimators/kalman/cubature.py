from filterpy_fixed.ckf import CubatureKalmanFilter as CKF

from dgsp.estimators.kalman.base import BaseKalmanFilter
from dgsp.functions import (
    dim_state,
    dim_observation,
)


class CubatureKalmanFilter(BaseKalmanFilter):
    kf: CKF

    def __init__(self, square_root: bool = False) -> None:
        super().__init__(
            CKF,
            lambda: dict(
                dt=self.dt,
                dim_x=dim_state,
                dim_z=dim_observation,
                fx=lambda x, dt: self.transition(x, dt),
                hx=lambda x: self.observation(x),
            ),
            square_root,
        )

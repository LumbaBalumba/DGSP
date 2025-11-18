from filterpy.kalman import UnscentedKalmanFilter as UKF, MerweScaledSigmaPoints

from dgsp.estimators.kalman.base import BaseKalmanFilter
from dgsp.functions import (
    dim_state,
    dim_observation,
)


class UnscentedKalmanFilter(BaseKalmanFilter):
    def __init__(self, square_root: bool = False) -> None:

        points = MerweScaledSigmaPoints(dim_state, alpha=0.1, beta=2.0, kappa=-1.0)

        super().__init__(
            UKF,
            lambda: dict(
                dt=self.dt,
                dim_x=dim_state,
                dim_z=dim_observation,
                fx=lambda x, dt: self.transition(x, dt),
                hx=lambda x: self.observation(x),
                points=points,
            ),
            square_root,
        )

from dgsp.estimators.kalman.unscented import (
    UnscentedKalmanFilter,
)


class ExtendedKalmanFilter(UnscentedKalmanFilter):
    def __init__(self, dt: float, square_root: bool = False, order: int = 1) -> None:
        super().__init__(dt, square_root)

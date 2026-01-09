from typing import Any
from typing_extensions import override

import numpy as np

from dgsp.estimators.base import Estimator


class BaseKalmanFilter(Estimator):
    kf: Any

    def __init__(
        self, kf_class: Any, kf_kwargs: Any, square_root: bool = False
    ) -> None:
        super().__init__()

        self.kf = kf_class(**kf_kwargs())
        self.kf.x = self.state[0]
        self.kf.P = self.P
        self.kf.Q = self.Q
        self.kf.R = self.R

    @override
    def predict(self, *args, **kwargs) -> None:
        self.kf.predict(*args, **kwargs)

        self.state.append(self.kf.x)
        self.k.append(self.kf.P)

        return super().predict()

    @override
    def update(self, data: np.ndarray, *args, **kwargs) -> None:
        self.kf.update(data, *args, **kwargs)

        self.state[-1] = self.kf.x
        self.k[-1] = self.kf.P.copy()

        return super().update(data)

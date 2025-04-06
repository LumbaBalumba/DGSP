from typing_extensions import override

import numpy as np

from dgsp.estimators.base import Estimator


class TrivialEstimator(Estimator):
    traj: np.ndarray
    var: np.ndarray

    def __init__(
        self,
        dt: float,
        all_traj: np.ndarray,
    ) -> None:
        super().__init__(dt)
        self.traj = np.mean(all_traj, axis=0)
        self.var = np.var(all_traj, axis=0)

    @override
    def predict(self) -> None:
        try:
            idx = int(np.ceil(self.time / self.dt))
            self.state.append(self.traj[idx, :])
            self.k.append(np.diag(self.var[idx, :]))
            super().predict()
        except IndexError:
            pass

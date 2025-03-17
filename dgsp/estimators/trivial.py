from typing_extensions import override

import numpy as np

from dgsp.estimators import Estimator


class TrivialEstimator(Estimator):
    traj: np.ndarray
    std: np.ndarray

    def __init__(
        self,
        dt: float,
        state: np.ndarray,
        k: np.ndarray,
        all_traj: np.ndarray,
    ) -> None:
        super().__init__(dt, state, k)
        self.traj = np.mean(all_traj, axis=0)
        self.std = np.std(all_traj, axis=0)

    @override
    def predict_step(self) -> None:
        idx = int(np.ceil(self.time / self.dt))
        self.state = self.traj[idx, :]
        self.k = self.std[idx, :]
        super().predict_step()

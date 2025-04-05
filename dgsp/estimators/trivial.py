from typing_extensions import override

import numpy as np

from dgsp.estimators.base import Estimator
from dgsp.functions import Q


class TrivialEstimator(Estimator):
    traj: np.ndarray
    std: np.ndarray

    def __init__(
        self,
        dt: float,
        all_traj: np.ndarray,
    ) -> None:
        super().__init__(dt, np.zeros(4), Q)
        self.traj = np.mean(all_traj, axis=0)
        self.std = np.std(all_traj, axis=0)

    @override
    def predict_step(self) -> None:
        try:
            idx = int(np.ceil(self.time / self.dt))
            self.state.append(self.traj[idx, :])
            self.k.append(np.diag(self.std[idx, :]))
            super().predict_step()
        except IndexError:
            pass

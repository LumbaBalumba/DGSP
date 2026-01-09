from typing_extensions import override

import numpy as np

from dgsp.estimators.base import Estimator
from scripts import dt_sim


class TrivialEstimator(Estimator):
    traj: np.ndarray
    var: np.ndarray

    def __init__(
        self,
        all_traj: np.ndarray,
    ) -> None:
        super().__init__()
        self.traj = np.mean(all_traj, axis=0)
        self.var = np.var(all_traj, axis=0)

    @override
    def predict(self) -> None:
        try:
            idx = int(np.ceil(self.time / dt_sim))
            self.state.append(self.traj[idx, :])
            self.k.append(np.diag(self.var[idx, :]))
            super().predict()
        except IndexError:
            pass

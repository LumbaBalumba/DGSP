import numpy as np

from dgsp.functions import initial, Q, R
from scripts import dt_sim, dt_pred


class Estimator:
    dt: float
    time: float
    state: list[np.ndarray]
    k: list[np.ndarray]
    Q: np.ndarray
    R: np.ndarray

    def __init__(
        self,
        dt: float,
    ) -> None:
        self.Q = Q * (dt_pred / dt_sim)
        self.R = R * (dt_pred / dt_sim)
        self.dt = dt
        self.state = [initial]
        self.k = [self.Q]
        self.time = 0.0

    def predict(self) -> None:
        self.time += self.dt

    def update(self, data: np.ndarray) -> None:
        pass

import numpy as np

from dgsp.functions import initial, Q
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
        self.Q = Q / dt_sim * dt_pred
        self.dt = dt
        self.state = [initial]
        self.k = [Q]
        self.time = 0.0

    def predict(self) -> None:
        self.time += self.dt

    def update(self, data: np.ndarray) -> None:
        pass

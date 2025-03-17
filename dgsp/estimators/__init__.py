import numpy as np


class Estimator:
    dt: float
    time: float
    state: np.ndarray
    k: np.ndarray

    def __init__(
        self,
        dt: float,
        state: np.ndarray,
        k: np.ndarray,
    ) -> None:
        self.dt = dt
        self.state = state
        self.k = k
        self.time = 0.0

    def predict_step(self) -> None:
        self.time += self.dt

    def correct_step(self, measurement: np.ndarray) -> None:
        self.time += self.dt

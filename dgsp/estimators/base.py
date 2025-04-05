import numpy as np

from dgsp.functions import initial, Q


class Estimator:
    dt: float
    time: float
    state: list[np.ndarray]
    k: list[np.ndarray]

    def __init__(
        self,
        dt: float,
    ) -> None:
        self.dt = dt
        self.state = [initial]
        self.k = [Q]
        self.time = 0.0

    def predict(self) -> None:
        self.time += self.dt

    def update(self, data: np.ndarray) -> None:
        self.time += self.dt

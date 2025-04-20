import numpy as np

from dgsp.functions import initial_guess, Q, R, P, observation_noise, transition_noise
from scripts import dt_sim, dt_pred


class Estimator:
    dt: float
    time: float
    state: list[np.ndarray]
    k: list[np.ndarray]
    Q: np.ndarray
    R: np.ndarray
    P: np.ndarray

    def __init__(
        self,
    ) -> None:
        self.dt = dt_pred
        self.Q = Q * (self.dt / dt_sim)
        self.R = R * (self.dt / dt_sim)
        self.P = P * (self.dt / dt_sim)
        self.state = [initial_guess]
        self.k = [self.P]
        self.time = 0.0

    def predict(self) -> None:
        self.time += self.dt

    def update(self, data: np.ndarray) -> None:
        pass

    def transition_noise(self) -> np.ndarray:
        return transition_noise(self.time) * dt_pred / dt_sim

    def observation_noise(self) -> np.ndarray:
        return observation_noise(self.time) * dt_pred / dt_sim

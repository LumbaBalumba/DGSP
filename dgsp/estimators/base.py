import numpy as np
import cupy as cp

from dgsp.functions import (
    initial_guess,
    Q,
    R,
    P,
    observation_noise,
    transition_noise,
    transition_cpu,
    observation_cpu,
    transition_gpu,
    observation_gpu,
    dim_state,
    dim_observation,
)
from scripts import dt_sim, dt_pred


class Estimator:
    dt: float
    time: float
    state: list[np.ndarray]
    k: list[np.ndarray]
    Q: np.ndarray
    R: np.ndarray
    P: np.ndarray
    backend_type: str = "numpy"

    def __init__(self, backend_type: str = "numpy") -> None:
        self.backend_type = backend_type
        backend = np if self.backend_type == "numpy" else cp

        self.dt = dt_pred
        self.Q = backend.asarray(Q) * (self.dt / dt_sim)
        self.R = backend.asarray(R) * (self.dt / dt_sim)
        self.P = backend.asarray(P)
        self.state = [backend.asarray(initial_guess)]
        self.k = [self.P]
        self.time = 0.0
        self.transition_func = (
            transition_cpu if self.backend_type == "numpy" else transition_gpu
        )
        self.observation_func = (
            observation_cpu if self.backend_type == "numpy" else observation_gpu
        )

    def predict(self) -> None:
        self.time += self.dt

    def update(self, data: np.ndarray) -> None:
        pass

    def transition_noise(self, size: int = 1) -> np.ndarray:
        return transition_noise(self.time, size, self.backend_type) * dt_pred / dt_sim

    def observation_noise(self, size: int = 1) -> np.ndarray:
        return observation_noise(self.time, size, self.backend_type) * dt_pred / dt_sim

    def transition(
        self,
        x: np.ndarray,
        batched: bool = False,
        dt: float | None = None,
    ) -> np.ndarray:
        if dt is None:
            dt = self.dt
        if batched:
            dxdt = (
                self.transition_func(x.T, self.time)
                .reshape((dim_state, -1))
                .T.reshape((-1, dim_state))
            )
        else:
            dxdt = self.transition_func(x, self.time)
        return dxdt * dt + x

    def observation(self, x: np.ndarray, batched: bool = False) -> np.ndarray:
        if batched:
            y = (
                self.observation_func(x.T, self.time)
                .reshape((dim_observation, -1))
                .T.reshape((-1, dim_observation))
            )
        else:
            y = self.observation_func(x, self.time)
        return y

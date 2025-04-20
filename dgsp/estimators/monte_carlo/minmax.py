"""
Conditionally‑Minimax Non‑linear Filter (CMNF) — этап 1.
Реализация совместима с абстрактным классом Estimator из DGSP.
Алгоритм: Kalman‑style «predict/update» + худшие Q,R (interval minimax).
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import inv

from dgsp.functions import (
    transition,
    transition_j,
    observation,
    observation_j,
    Q as Q_nom,
    R as R_nom,
    initial_guess,
)
from scripts import dt_sim, dt_pred
from dgsp.estimators.base import Estimator


class MinMaxFilter(Estimator):

    def __init__(
        self,
        dt: float,
        q_scale: float = 2.0,
        r_scale: float = 2.0,
    ) -> None:
        super().__init__(dt)

        self.Q_worst = Q_nom * q_scale * (dt_pred / dt_sim)
        self.R_worst = R_nom * r_scale * (dt_pred / dt_sim)
        self.x = initial_guess

    def predict(self) -> None:
        F = transition_j(self.x, self.time)
        self.x = self.x + transition(self.x, self.time) * self.dt

        self.P = F @ self.P @ F.T + self.Q_worst
        self.state.append(self.x.copy())
        self.k.append(self.P.copy())

        super().predict()

    def update(self, data: np.ndarray) -> None:
        H = observation_j(self.x, self.time)
        S = H @ self.P @ H.T + self.R_worst
        K = self.P @ H.T @ inv(S)

        y = data - observation(self.x, self.time)
        self.x += K @ y
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P

        self.state[-1] = self.x.copy()
        self.k[-1] = self.P.copy()

        super().update(data)

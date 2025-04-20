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
from dgsp.estimators.base import Estimator  # путь, который есть в вашем проекте


class MinMaxFilter(Estimator):
    """Несмещённый робастный фильтр (conditionally‑minimax)."""

    def __init__(
        self,
        dt: float,
        q_scale: float = 2.0,  # множитель неопределённости Q
        r_scale: float = 2.0,  # множитель неопределённости R
    ) -> None:
        super().__init__(dt)

        # «худшие» ковариации, пересчитанные на дискретный шаг dt_pred
        self.Q_worst = Q_nom * q_scale * (dt_pred / dt_sim)
        self.R_worst = R_nom * r_scale * (dt_pred / dt_sim)
        self.x = initial_guess

    # --------------------------------------------------------------------- #
    #  PREDICT
    # --------------------------------------------------------------------- #
    def predict(self) -> None:
        """
        Шаг прогноза CMNF (identical to EKF, но с Q_worst).
        """
        F = transition_j(self.x, self.time)  # Якобиан f(x)
        self.x = self.x + transition(self.x, self.time) * self.dt

        self.P = F @ self.P @ F.T + self.Q_worst  # «самый плохой» Q
        self.state.append(self.x.copy())
        self.k.append(self.P.copy())

        super().predict()  # увеличиваем self.time

    # --------------------------------------------------------------------- #
    #  UPDATE
    # --------------------------------------------------------------------- #
    def update(self, data: np.ndarray) -> None:
        """
        Шаг коррекции: Kalman‑gain на «худшей» инновации.
        Для этапа 1 берём K = P Hᵀ (H P Hᵀ + R*)⁻¹.
        """
        H = observation_j(self.x, self.time)  # Якобиан h(x)
        S = H @ self.P @ H.T + self.R_worst  # «плохая» инновация
        K = self.P @ H.T @ inv(S)  # minimax‑gain (пока без LMI)

        y = data - observation(self.x, self.time)  # инновация
        self.x += K @ y
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P

        # сохранить результат шага
        self.state[-1] = self.x.copy()
        self.k[-1] = self.P.copy()

        super().update(data)

import numpy as np

from dgsp.functions import (
    transition,
    transition_noise,
    measurement,
    measurement_noise,
    initial,
)


class RobotSystem:
    trajectory: np.ndarray
    time: float
    dt: float
    t_max: float
    init: np.ndarray = np.array(initial, dtype=float)

    def __init__(self, dt: float, t_max: float) -> None:
        self.trajectory = np.random.normal(self.init, 0.3 * np.ones(4)).reshape((-1, 4))
        self.measurements = measurement(self.trajectory[0]).reshape((-1, 2))
        self.time = 0.0
        self.dt = dt
        self.t_max = t_max

    def step(self) -> None:
        old_state = self.trajectory[-1]
        new_state = old_state + transition(old_state) * self.dt + transition_noise()
        new_measurement = measurement(new_state) + measurement_noise()
        self.trajectory = np.append(self.trajectory, [new_state], axis=0)
        self.measurements = np.append(self.measurements, [new_measurement], axis=0)
        self.time += self.dt

    def simulate(self) -> None:
        n = int(np.ceil(self.t_max / self.dt))
        for _ in range(n):
            self.step()

    def save(self, traj_filename: str, meas_filename: str) -> None:
        np.save(traj_filename, self.trajectory)
        np.save(meas_filename, self.measurements)

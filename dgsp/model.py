import numpy as np

from dgsp.functions import (
    transition_cpu,
    transition_noise,
    observation_cpu,
    observation_noise,
    initial,
)


class RobotSystem:
    trajectory: list[np.ndarray]
    observations: list[np.ndarray]
    time: float
    dt: float
    t_max: float
    init: np.ndarray = np.array(initial, dtype=float)

    def __init__(self, dt: float, t_max: float) -> None:
        self.trajectory = [initial]
        self.time = 0.0
        self.observations = [observation_cpu(self.trajectory[0], self.time)]
        self.dt = dt
        self.t_max = t_max

    def step(self) -> None:
        old_state = self.trajectory[-1]
        new_state = (
            old_state
            + transition_cpu(old_state, self.time) * self.dt
            + transition_noise(self.time, dt=self.dt)
        )
        new_observation = observation_cpu(new_state, self.time) + observation_noise(
            self.time, dt=self.dt
        )
        self.trajectory.append(new_state)
        self.observations.append(new_observation)
        self.time += self.dt

    def simulate(self) -> None:
        n = int(np.ceil(self.t_max / self.dt))
        for _ in range(n):
            self.step()

    def save(self, traj_filename: str, obs_filename: str) -> None:
        np.save(traj_filename, self.trajectory)
        np.save(obs_filename, self.observations)

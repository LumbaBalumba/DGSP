import numpy as np
import sympy as sp

x1, x2, theta, phi = sp.symbols(r"x_1 x_2 \theta \phi")
x = sp.Matrix([x1, x2, theta, phi])

T_MAX = 10.0
dt = 0.0025

l = 0.1
Q = np.eye(2) * 0.3
R = np.eye(2) * 0.2
u1 = 3.0
u2 = 3.0


sp_transition = sp.Matrix(
    [
        sp.cos(theta) * sp.cos(phi) * u1,
        sp.sin(theta) * sp.sin(phi) * u1,
        sp.sin(phi) * u1 / l,
        u2,
    ]
)
transition = sp.lambdify(x, sp_transition)

sp_measurement = sp.Matrix([(x1**2 + x2**2) ** 0.5, sp.atan(x2 / x1)])
measurement = sp.lambdify(x, sp_measurement)


def transition_noise() -> np.ndarray:
    return np.concatenate(
        [[0, 0], np.random.multivariate_normal(np.zeros(2), Q)], axis=0
    )


def measurement_noise():
    return np.random.multivariate_normal(np.zeros(2), R)


class RobotSystem:
    trajectory: np.ndarray
    time: float

    def __init__(self) -> None:
        self.trajectory = np.random.normal(np.zeros(4), 0.3 * np.ones(4)).reshape(
            (-1, 4)
        )
        self.measurements = measurement(*self.trajectory[0]).reshape((-1, 2))
        self.time = 0.0

    def step(self) -> None:
        old_state = self.trajectory[-1]
        new_state = (
            old_state + transition(*old_state).reshape((-1)) * dt + transition_noise()
        )
        new_measurement = measurement(*new_state)[0] + measurement_noise()
        self.trajectory = np.append(self.trajectory, [new_state], axis=0)
        self.measurements = np.append(self.measurements, [new_measurement], axis=0)
        self.time += dt

    def simulate(self) -> None:
        n = int(np.ceil(T_MAX / dt))
        for _ in range(n):
            self.step()

    def save(self, traj_filename: str, meas_filename: str) -> None:
        np.save(traj_filename, self.trajectory)
        np.save(meas_filename, self.measurements)

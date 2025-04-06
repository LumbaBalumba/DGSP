import numpy as np
import sympy as sp
import numba


x1, x2, theta, phi = sp.symbols(r"x_1 x_2 \theta \phi")
x = sp.Matrix([x1, x2, theta, phi])


l = 0.1
Q = np.diag([3e-3, 3e-3, 3e-1, 3e-1]) / 1e4
R = np.eye(2) / 1e3
u1 = 3.0
u2 = 1e-1 / 20 * 0

initial = np.array([0.0, 0.0, 0.0, 0.0])


sp_transition = sp.Matrix(
    [
        sp.cos(theta) * sp.cos(phi) * u1,
        sp.sin(theta) * sp.cos(phi) * u1,
        sp.sin(phi) * u1 / l,
        u2,
    ]
)
sp_measurement = sp.Matrix([(x1**2 + x2**2) ** 0.5, sp.atan2(x2, x1)])

dim_state = sp_transition.shape[0]
dim_measurement = sp_measurement.shape[0]

transition_c = numba.njit()(sp.lambdify(x, sp_transition))
transition = lambda x: transition_c(*x).reshape((-1))

measurement_c = numba.njit()(sp.lambdify(x, sp_measurement))
measurement = lambda x: measurement_c(*x).reshape((-1))


@numba.njit()
def transition_noise() -> np.ndarray:
    noise = np.empty(dim_state)
    for i in range(dim_state):
        noise[i] = np.random.normal(0, Q[i, i] ** 0.5)
    return noise


@numba.njit()
def measurement_noise() -> np.ndarray:
    noise = np.empty(dim_measurement)
    for i in range(dim_measurement):
        noise[i] = np.random.normal(0, R[i, i] ** 0.5)
    return noise

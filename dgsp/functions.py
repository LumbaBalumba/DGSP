import numpy as np
import sympy as sp
import numba


x1, x2, theta, phi = sp.symbols(r"x_1 x_2 \theta \phi")
x = sp.Matrix([x1, x2, theta, phi])


l = 0.1
Q = np.diag([3e-3, 3e-3, 3e-1, 3e-1])
R = np.eye(2) * 0.2
u1 = 3.0
u2 = 3.0

sp_transition = sp.Matrix(
    [
        sp.cos(theta) * sp.cos(phi) * u1,
        sp.sin(theta) * sp.cos(phi) * u1,
        sp.sin(phi) * u1 / l,
        u2,
    ]
)
transition_c = numba.jit(nopython=True)(sp.lambdify(x, sp_transition))
transition = lambda x: transition_c(*x).reshape((-1))
transition_J = sp.lambdify(x, sp_transition.jacobian(x))

sp_measurement = sp.Matrix([(x1**2 + x2**2) ** 0.5, sp.atan2(x2, x1)])
measurement_c = numba.jit(nopython=True)(sp.lambdify(x, sp_measurement))
measurement = lambda x: measurement_c(*x).reshape((-1))
measurement_J = sp.lambdify(x, sp_measurement.jacobian(x))


@numba.jit(nopython=True)
def transition_noise() -> np.ndarray:
    noise = np.empty(4)
    for i in range(4):
        noise[i] = np.random.normal(0, Q[i, i] ** 0.5)
    return noise


@numba.jit(nopython=True)
def measurement_noise() -> np.ndarray:
    noise = np.empty(2)
    for i in range(2):
        noise[i] = np.random.normal(0, R[i, i] ** 0.5)
    return noise

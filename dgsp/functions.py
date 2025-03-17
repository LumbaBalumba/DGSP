import numpy as np
import sympy as sp
import numba


x1, x2, theta, phi = sp.symbols(r"x_1 x_2 \theta \phi")
x = sp.Matrix([x1, x2, theta, phi])


l = 0.1
Q = np.diag([0, 0, 0.3, 0.3])
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
transition = numba.jit(nopython=True)(sp.lambdify(x, sp_transition))
transition_J = sp.lambdify(x, sp_transition.jacobian(x))

sp_measurement = sp.Matrix([(x1**2 + x2**2) ** 0.5, sp.atan(x2 / x1)])
measurement = numba.jit(nopython=True)(sp.lambdify(x, sp_measurement))
measurement_J = sp.lambdify(x, sp_measurement.jacobian(x))


@numba.jit(nopython=True)
def transition_noise() -> np.ndarray:
    noise = np.empty(4)
    for i in range(4):
        noise[i] = np.random.normal(0, Q[i, i])
    return noise


@numba.jit(nopython=True)
def measurement_noise() -> np.ndarray:
    noise = np.empty(2)
    for i in range(2):
        noise[i] = np.random.normal(0, R[i, i])
    return noise

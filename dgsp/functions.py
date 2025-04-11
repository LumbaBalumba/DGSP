import numpy as np
import sympy as sp
import numba


x1, x2, theta, phi = sp.symbols(r"x_1 x_2 \theta \phi")
t = sp.Symbol("t")
x = sp.Matrix([x1, x2, theta, phi])


Q = np.diag([3e-3, 3e-3, 3e-1, 3e-1]) / 1e4
R = np.eye(2) / 1e3 * 5

l = 0.1
u1 = 3.0
u2 = 0.0

initial = np.array([0.0, 0.0, np.pi / 4, 0.0])


sp_transition = sp.Matrix(
    [
        sp.cos(theta) * sp.cos(phi) * u1,
        sp.sin(theta) * sp.cos(phi) * u1,
        sp.sin(phi) * u1 / l,
        u2,
    ]
)
sp_transition_j = sp_transition.jacobian(x)

sp_observation = sp.Matrix([(x1**2 + x2**2) ** 0.5, sp.atan2(x2, x1)])
sp_observation_j = sp_observation.jacobian(x)

dim_state = sp_transition.shape[0]
dim_observation = sp_observation.shape[0]


def prettify(func):
    func_c = numba.njit()(sp.lambdify([*x, t], func))
    return lambda x, t: func_c(*x, t).reshape((-1))


transition = prettify(sp_transition)
transition_j = prettify(sp_transition_j)

observation = prettify(sp_observation)
observation_j = prettify(sp_observation_j)


@numba.njit()
def transition_noise() -> np.ndarray:
    noise = np.empty(dim_state)
    for i in range(dim_state):
        noise[i] = np.random.normal(0, Q[i, i] ** 0.5)
    return noise


@numba.njit()
def observation_noise() -> np.ndarray:
    noise = np.empty(dim_observation)
    for i in range(dim_observation):
        noise[i] = np.random.normal(0, R[i, i] ** 0.5)
    return noise

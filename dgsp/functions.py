import numpy as np
import sympy as sp
import cupy as cp
import warnings

warnings.filterwarnings("ignore")

x = sp.IndexedBase("x")
w = sp.IndexedBase("w")
t = sp.Symbol("t")

# Код для редактирования под вашу статью начинается здесь
###########################################################################
dim_state = 4
dim_observation = 2

Q = np.diag(np.array([3e-3, 3e-3, 3e-1, 3e-1])) / 1e4
P = np.diag([3e-1, 3e-1, 3e-1, 3e-1]) / 1e6
R = np.eye(2) / 1e3 * 5

L = 0.1

u1 = 3.0
u2 = 0.0


def transition(x, w):
    return [
        sp.cos(x[2]) * sp.cos(x[3]) * u1 + w[0],
        sp.sin(x[2]) * sp.cos(x[3]) * u1 + w[1],
        sp.sin(x[3]) * u1 / L + w[2],
        u2 + w[3],
    ]


def observation(x, w):
    return [(x[0] ** 2 + x[1] ** 2) ** 0.5 + w[0], sp.atan2(x[1], x[0]) + w[1]]


initial = np.array([0.0, 0.0, np.pi / 4, 0.0])
initial_guess = initial
###########################################################################

sp_transition = sp.Matrix(transition(x, w))
sp_transition_j = sp_transition.jacobian(
    [x[i] for i in range(dim_state)] + [w[i] for i in range(dim_state)]
)

sp_observation = sp.Matrix(observation(x, w))
sp_observation_j = sp_observation.jacobian(
    [x[i] for i in range(dim_state)] + [w[i] for i in range(dim_state)]
)


def prettify(func, backend_type="numpy"):
    F_raw = sp.lambdify((x, w, t), func, backend_type)
    return lambda x, w, t: F_raw(x, w, t).reshape(-1)


transition_cpu = prettify(sp_transition)
transition_cpu_j = prettify(sp_transition_j)

transition_gpu = prettify(sp_transition, "cupy")
transition_gpu_j = prettify(sp_transition_j, "cupy")

observation_cpu = prettify(sp_observation)
observation_cpu_j = prettify(sp_observation_j)

observation_gpu = prettify(sp_observation, "cupy")
observation_gpu_j = prettify(sp_observation_j, "cupy")


def transition_noise(
    t: float, size: int = 1, backend_type: str = "numpy"
) -> np.ndarray:
    backend = np if backend_type == "numpy" else cp

    if size == 1:
        return backend.random.multivariate_normal(
            mean=backend.zeros(dim_state), cov=backend.asarray(Q)
        )
    else:
        return backend.random.multivariate_normal(
            mean=backend.zeros(dim_state), cov=backend.asarray(Q), size=size
        )


def observation_noise(
    t: float, size: int = 1, backend_type: str = "numpy"
) -> np.ndarray:
    backend = np if backend_type == "numpy" else cp

    if size == 1:
        return backend.random.multivariate_normal(
            mean=backend.zeros(dim_observation), cov=backend.asarray(R)
        )
    else:
        return backend.random.multivariate_normal(
            mean=backend.zeros(dim_observation), cov=backend.asarray(R), size=size
        )

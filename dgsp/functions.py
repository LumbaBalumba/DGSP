import numpy as np
import sympy as sp
import cupy as cp
import warnings

warnings.filterwarnings("ignore")

x = sp.IndexedBase("x")
t = sp.Symbol("t")

# Код для редактирования под вашу статью начинается здесь
###########################################################################
dim_state = 6
dim_observation = 2

Q = np.diag(np.array([3e-3, 3e-3, 3e-1, 3e-1, 0, 0])) / 1e4
P = np.diag([3e-1, 3e-1, 3e-1, 3e-1, 0.0, 0.0])
R = np.eye(2) / 1e3 * 5

l = 0.1


def transition(x):
    return [
        sp.cos(x[2]) * sp.cos(x[3]) * x[4],
        sp.sin(x[2]) * sp.cos(x[3]) * x[4],
        sp.sin(x[3]) * x[4] / l,
        x[5],
        x[4],
        x[5],
    ]


def observation(x):
    return [(x[0] ** 2 + x[1] ** 2) ** 0.5, sp.atan2(x[1], x[0])]


initial = np.array([0.0, 0.0, np.pi / 4, 0.0, 3.0, 0.0])
initial_guess = initial
###########################################################################

sp_transition = sp.Matrix(transition(x))

sp_transition_j = sp_transition.jacobian([x[i] for i in range(dim_state)])

sp_observation = sp.Matrix(observation(x))
sp_observation_j = sp_observation.jacobian([x[i] for i in range(dim_state)])


def prettify(func, backend="numpy"):
    F_raw = sp.lambdify((x, t), func, backend)
    return lambda x, t: F_raw(x, t).reshape(-1)


transition_cpu = prettify(sp_transition)
transition_cpu_j = prettify(sp_transition_j)

transition_gpu = prettify(sp_transition, "cupy")
transition_gpu_j = prettify(sp_transition_j, "cupy")

observation_cpu = prettify(sp_observation)
observation_cpu_j = prettify(sp_observation_j)

observation_gpu = prettify(sp_observation, "cupy")
observation_gpu_j = prettify(sp_observation_j, "cupy")


def transition_noise(t: float, backend_type: str = "numpy") -> np.ndarray:
    backend = np if backend_type == "numpy" else cp
    noise = backend.random.multivariate_normal(
        mean=backend.zeros(dim_state), cov=backend.asarray(Q)
    )
    return noise


def observation_noise(t: float, backend_type: str = "numpy") -> np.ndarray:
    backend = np if backend_type == "numpy" else cp
    noise = backend.random.multivariate_normal(
        mean=backend.zeros(dim_observation), cov=backend.asarray(R)
    )
    return noise

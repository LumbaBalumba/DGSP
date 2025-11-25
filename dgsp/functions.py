import numpy as np
import sympy as sp
import cupy as cp
import warnings

warnings.filterwarnings("ignore")

x = sp.IndexedBase("x")
placeholder = sp.IndexedBase("w")
t = sp.Symbol("t")

# Код для редактирования под вашу статью начинается здесь
###########################################################################
dim_state = 4
dim_observation = 2

Q = np.diag(np.array([3e-3, 3e-3, 3e-1, 3e-1])) / 1e2
P = np.diag([3e-3, 3e-3, 3e-1, 3e-1]) / 1e2
R = np.eye(2) / 1e3 * 5

L = 0.1

u1 = 3.0
u2 = 3.0


def transition(x):
    return [
        sp.cos(x[2]) * sp.cos(x[3]) * u1,
        sp.sin(x[2]) * sp.cos(x[3]) * u1,
        sp.sin(x[3]) * u1 / L,
        u2,
    ]


def observation(x):
    return [(x[0] ** 2 + x[1] ** 2) ** 0.5, sp.atan2(x[1], x[0])]


initial = np.array([0.0, 0.0, np.pi / 4, 0.0])
initial_guess = initial
###########################################################################

sp_transition = sp.Matrix(transition(x)) + sp.Matrix(
    [placeholder[i] for i in range(dim_state)]
)
sp_transition_j = sp_transition.jacobian([x[i] for i in range(dim_state)])

sp_observation = sp.Matrix(observation(x)) + sp.Matrix(
    [placeholder[i] for i in range(dim_observation)]
)
sp_observation_j = sp_observation.jacobian([x[i] for i in range(dim_state)])


def prettify(func, backend_type="numpy", flatten=True):
    F_raw = sp.lambdify((x, placeholder, t), func, backend_type)

    backend = np if backend_type == "numpy" else cp

    def obj(x, t):
        res = F_raw(x, backend.zeros_like(x), t)

        if flatten:
            res = res.reshape(-1)

        return res

    return obj


transition_cpu = prettify(sp_transition)
transition_cpu_j = prettify(sp_transition_j, flatten=False)

transition_gpu = prettify(sp_transition, "cupy")
transition_gpu_j = prettify(sp_transition_j, "cupy", flatten=False)

observation_cpu = prettify(sp_observation)
observation_cpu_j = prettify(sp_observation_j, flatten=False)

observation_gpu = prettify(sp_observation, "cupy")
observation_gpu_j = prettify(sp_observation_j, "cupy", flatten=False)


def transition_noise(
    t: float, size: int = 1, backend_type: str = "numpy", dt: float = 1
) -> np.ndarray:
    backend = np if backend_type == "numpy" else cp

    if size == 1:
        return backend.random.multivariate_normal(
            mean=backend.zeros(dim_state), cov=backend.asarray(Q * np.sqrt(dt))
        )
    else:
        return backend.random.multivariate_normal(
            mean=backend.zeros(dim_state),
            cov=backend.asarray(Q * np.sqrt(dt)),
            size=size,
        )


def observation_noise(
    t: float, size: int = 1, backend_type: str = "numpy", dt: float = 1
) -> np.ndarray:
    backend = np if backend_type == "numpy" else cp

    if size == 1:
        return backend.random.multivariate_normal(
            mean=backend.zeros(dim_observation), cov=backend.asarray(R * dt)
        )
    else:
        return backend.random.multivariate_normal(
            mean=backend.zeros(dim_observation),
            cov=backend.asarray(R * dt),
            size=size,
        )

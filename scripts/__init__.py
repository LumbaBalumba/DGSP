T_MAX = 1e0
dt_sim = 1e-4
dt_pred = dt_sim * 10
dt_obs = dt_pred * 10

NUM_TRAJECTORIES = 100

ESTIMATORS = [
    "ekf",
    # "ekfr",
    "ekf2",
    # "ekf2r",
    # "ckf",
    # "ckfr",
    # "ukf",
    # "ukfr",
    # "pf",
    # "pfb",
    # "cmnf",
    # "trivial",
]
MONTE_CARLO_BACKEND = "numpy"  # "numpy" or "cupy"
MONTE_CARLO_NUM_PARTICLES = 1000

CONVERGENCE_COMPONENTS_CHECK = [0, 1]  # 'all' or list of state components
CONVERGENCE_RATIO = 0.9

ENABLE_PARALLEL = True

T_MAX = 1e0
dt_sim = 1e-3
dt_pred = dt_sim
dt_obs = dt_pred * 10

NUM_TRAJECTORIES = 10000

ESTIMATORS = [
    "ekf",
    # "ekfr",
    # "ekf2",
    # "ekf2r",
    "ckf",
    # "ckfr",
    "ukf",
    # "ukfr",
    "pf",
    # "pfb",
    # "cmnf",
    "trivial",
]
MONTE_CARLO_BACKEND = "numpy"  # "numpy" or "cupy"
MONTE_CARLO_NUM_PARTICLES = 1000

ENABLE_PARALLEL = True

T_MAX = 1e0
dt_sim = 1e-4
dt_pred = dt_sim
dt_obs = dt_pred * 10

NUM_TRAJECTORIES = 1

ESTIMATORS = [
    # "ekf",
    # "ekfr",
    # "ckf",
    # "ckfr",
    "ukf",
    # "ukfr",
    "pf",
    # "pfb",
    # "cmnf",
    # "trivial",
]
MONTE_CARLO_BACKEND = "cupy"  # "numpy" or "cupy"
MONTE_CARLO_NUM_PARTICLES = 1000

ENABLE_PARALLEL = False

T_MAX = 1e0
dt_sim = 1e-4
dt_pred = dt_sim
dt_obs = dt_pred * 10

NUM_TRAJECTORIES = 1000

ESTIMATORS = [
    "ekf",
    "ekfr",
    "ckf",
    "ckfr",
    "ukf",
    "ukfr",
    "pf",
    "pfb",
    "cmnf",
    "trivial",
]

ENABLE_PARALLEL = True

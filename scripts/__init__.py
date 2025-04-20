T_MAX = 1e0
dt_sim = 1e-4
dt_pred = dt_sim
dt_obs = dt_pred * 10

NUM_TRAJECTORIES = 10

ESTIMATORS = [
    "trivial",
    "ekf",
    "ekfr",
    "ekf2",
    "ekf2r",
    "ckf",
    "ckfr",
    "ukf",
    "ukfr",
    "pf",
    "pfb",
]

ENABLE_PARALLEL = True

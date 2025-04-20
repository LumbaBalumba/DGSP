T_MAX = 1e0
dt_sim = 1e-4
dt_pred = dt_sim
dt_obs = dt_pred * 10

NUM_TRAJECTORIES = 1000

ESTIMATORS = ["ckf", "ukf", "trivial"]

ENABLE_PARALLEL = True

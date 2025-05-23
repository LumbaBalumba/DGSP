import os
from copy import deepcopy

from joblib import Parallel, delayed
import numpy as np

from dgsp.estimators import (
    Estimator,
    TrivialEstimator,
    ExtendedKalmanFilter,
    CubatureKalmanFilter,
    UnscentedKalmanFilter,
    ParticleFilter,
    MinMaxFilter,
)
from scripts import (
    ENABLE_PARALLEL,
    ESTIMATORS,
    dt_pred,
    dt_obs,
    dt_sim,
    NUM_TRAJECTORIES,
)


def estimate_one(traj_n: int, estimator: Estimator, estimator_dir: str) -> None:
    obs = np.load(os.path.join("data", "obs", f"{traj_n}.npy"))

    pred_step = int(dt_pred / dt_sim)
    correct_step = int(dt_obs / dt_sim)

    for i in range(0, len(obs), pred_step):
        estimator.predict()
        if i % correct_step == 0:
            estimator.update(obs[i])

    traj_est, k_est = estimator.state, estimator.k

    new_path_dir = os.path.join("data", "estimate", estimator_dir)
    new_path_traj = os.path.join(new_path_dir, "traj")
    new_path_k = os.path.join(new_path_dir, "k")

    if not os.path.exists(new_path_traj):
        os.makedirs(new_path_traj)
    np.save(os.path.join(new_path_traj, f"{traj_n}.npy"), traj_est)

    if not os.path.exists(new_path_k):
        os.makedirs(new_path_k)
    np.save(os.path.join(new_path_k, f"{traj_n}.npy"), k_est)


def estimate_all(estimator_type: str, parallel: bool = True) -> None:
    match estimator_type:
        case "ekf":
            estimator = ExtendedKalmanFilter()
        case "ekfr":
            estimator = ExtendedKalmanFilter(square_root=True)
        case "ekf2":
            estimator = ExtendedKalmanFilter(order=2)
        case "ekf2r":
            estimator = ExtendedKalmanFilter(order=2, square_root=True)
        case "ukf":
            estimator = UnscentedKalmanFilter()
        case "ukfr":
            estimator = UnscentedKalmanFilter(square_root=True)
        case "ckf":
            estimator = CubatureKalmanFilter()
        case "ckfr":
            estimator = CubatureKalmanFilter(square_root=True)
        case "trivial":
            all_traj = [
                np.load(os.path.join("data", "traj", f"{i}.npy"))
                for i in range(NUM_TRAJECTORIES)
            ]
            estimator = TrivialEstimator(np.array(all_traj))
        case "pf":
            estimator = ParticleFilter(1000)
        case "pfb":
            estimator = ParticleFilter(1000, bootstrap=True)
        case "cmnf":
            estimator = MinMaxFilter(1000)
        case _:
            raise RuntimeError(f"Invalid estimator type: {estimator_type}")

    if parallel:
        Parallel(n_jobs=-1, verbose=10)(
            delayed(estimate_one)(idx, deepcopy(estimator), estimator_type)
            for idx in range(NUM_TRAJECTORIES)
        )
    else:
        for idx in range(NUM_TRAJECTORIES):
            estimate_one(idx, deepcopy(estimator), estimator_type)


def estimate(parallel: bool = ENABLE_PARALLEL) -> None:
    for estimator_type in ESTIMATORS:
        print(f"Running {estimator_type} estimator")
        estimate_all(estimator_type, parallel)

from copy import deepcopy
import os

import numpy as np

from dgsp.estimators.base import Estimator
from dgsp.estimators.trivial import TrivialEstimator
from dgsp.estimators.unscented import UnscentedKalmanFilter
from scripts import dt_pred, dt_meas, dt_sim
from scripts.generate import NUM_TRAJECTORIES


def estimate_one(traj_n: int, estimator: Estimator) -> None:
    meas = np.load(f"./data/meas/{traj_n}.npy")

    pred_step = int(dt_pred / dt_sim)
    correct_step = int(dt_meas / dt_sim)

    for i in range(0, len(meas), pred_step):
        estimator.predict_step()
        if i % correct_step == 0:
            estimator.correct_step(meas[i])

    traj_est, k_est = estimator.state, estimator.k

    new_path_dir = os.path.join("data", "estimate", "ukf")
    new_path_traj = os.path.join(new_path_dir, "traj")
    new_path_k = os.path.join(new_path_dir, "k")

    if not os.path.exists(new_path_traj):
        os.makedirs(new_path_traj)
        np.save(os.path.join(new_path_traj, f"{traj_n}.npy"), traj_est)

    if not os.path.exists(new_path_k):
        os.makedirs(new_path_k)
        np.save(os.path.join(new_path_k, f"{traj_n}.npy"), k_est)


def estimate_all(estimator_type: str) -> None:
    match estimator_type:
        case "UKF":
            estimator = UnscentedKalmanFilter(dt_pred)
        case "trivial":
            all_traj = [
                np.load(f"./data/traj/{i}.npy") for i in range(NUM_TRAJECTORIES)
            ]
            estimator = TrivialEstimator(dt_pred, np.array(all_traj))
        case _:
            raise RuntimeError(f"Invalid estimator type: {estimator_type}")

    for idx in range(NUM_TRAJECTORIES):
        estimate_one(idx, deepcopy(estimator))


def estimate():
    for estimator_type in ["UKF", "UKFR", "PF", "trivial"]:
        estimate_all(estimator_type)

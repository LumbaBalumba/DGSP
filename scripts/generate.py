import os

from joblib import Parallel, delayed
import numpy as np

import dgsp.model as model
from scripts import dt_sim, T_MAX, NUM_TRAJECTORIES


np.random.seed(100)

if not os.path.exists("data/traj"):
    os.makedirs("data/traj")

if not os.path.exists("data/obs"):
    os.makedirs("data/obs")


def generate_one(idx: int) -> None:
    system = model.RobotSystem(dt=dt_sim, t_max=T_MAX)
    system.simulate()
    traj_filename = os.path.join("data", "traj", f"{idx}.npy")
    meas_filename = os.path.join("data", "obs", f"{idx}.npy")
    system.save(traj_filename, meas_filename)


def generate_all(parallel: bool = True) -> None:
    if parallel:
        Parallel(n_jobs=-1, verbose=10)(
            delayed(generate_one)(i) for i in range(NUM_TRAJECTORIES)
        )
    else:
        [generate_one(i) for i in range(NUM_TRAJECTORIES)]

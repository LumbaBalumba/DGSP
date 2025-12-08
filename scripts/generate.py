import os

from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

import dgsp.model as model
from scripts import ENABLE_PARALLEL, dt_sim, T_MAX, NUM_TRAJECTORIES


def generate_one(idx: int) -> None:
    if not os.path.exists(os.path.join("data", "traj")):
        os.makedirs(os.path.join("data", "traj"))

    if not os.path.exists(os.path.join("data", "obs")):
        os.makedirs(os.path.join("data", "obs"))

    system = model.RobotSystem(dt=dt_sim, t_max=T_MAX)
    system.simulate()
    traj_filename = os.path.join("data", "traj", f"{idx}.npy")
    obs_filename = os.path.join("data", "obs", f"{idx}.npy")
    system.save(traj_filename, obs_filename)


def generate_all(parallel: bool = ENABLE_PARALLEL) -> None:
    if parallel:
        with tqdm_joblib(desc="Generation", total=NUM_TRAJECTORIES):
            Parallel(n_jobs=-1, batch_size=100)(
                delayed(generate_one)(i) for i in range(NUM_TRAJECTORIES)
            )
    else:
        [generate_one(i) for i in tqdm(range(NUM_TRAJECTORIES))]

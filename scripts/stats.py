import os

import numpy as np
import pandas as pd

from dgsp.functions import dim_state
from scripts import ESTIMATORS, T_MAX, dt_pred


def stats() -> None:
    example_traj_num = 0

    traj = np.load(os.path.join("data", "traj", f"{example_traj_num}.npy"))

    t = np.linspace(0, T_MAX, int(np.ceil(T_MAX / dt_pred)))
    traj = traj[:: len(traj) // len(t)][: len(t)]
    traj_estimates = [
        np.load(
            os.path.join(
                "data",
                "estimate",
                f"{estimator}",
                "traj",
                f"{example_traj_num}.npy",
            )
        )
        for estimator in ESTIMATORS
    ]
    k_estimates = [
        np.load(
            os.path.join(
                "data", "estimate", f"{estimator}", "k", f"{example_traj_num}.npy"
            )
        )
        for estimator in ESTIMATORS
    ]

    if not os.path.exists("stats"):
        os.makedirs("stats")

    for component in range(dim_state):
        x = traj[:, component]
        x_est = [traj_est[:, component][: len(x)] for traj_est in traj_estimates]
        std_est = [k_est[:, component, component][: len(x)] for k_est in k_estimates]

        df_x = pd.DataFrame(
            {
                "t": t,
                "x": x,
                **{estimator: est for estimator, est in zip(ESTIMATORS, x_est)},
            }
        )
        df_k = pd.DataFrame(
            {
                "t": t,
                "x": x,
                **{estimator: est for estimator, est in zip(ESTIMATORS, std_est)},
            }
        )
        df_x.to_csv(os.path.join("stats", f"example_x_{component}.csv"), index=False)
        df_k.to_csv(os.path.join("stats", f"example_k_{component}.csv"), index=False)

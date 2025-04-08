import os

import numpy as np
import pandas as pd

from dgsp.functions import dim_state
from scripts import T_MAX, dt_pred


def stats() -> None:
    example_traj_num = 0

    traj = np.load(f"./data/traj/{example_traj_num}.npy")

    estimators = ["trivial", "ukf", "ukfr", "pf"]

    t = np.linspace(0, T_MAX, int(np.ceil(T_MAX / dt_pred)))
    traj = traj[:: len(traj) // len(t)][: len(t)]
    traj_estimates = [
        np.load(f"./data/estimate/{estimator}/traj/{example_traj_num}.npy")
        for estimator in estimators
    ]
    k_estimates = [
        np.load(f"./data/estimate/{estimator}/k/{example_traj_num}.npy")
        for estimator in estimators
    ]

    if not os.path.exists("./stats"):
        os.makedirs("./stats")

    for component in range(dim_state):
        x = traj[:, component]
        x_est = [traj_est[:, component] for traj_est in traj_estimates]
        std_est = [k_est[:, component, component] for k_est in k_estimates]
        print(t.shape, x.shape)
        df_x = pd.DataFrame(
            {
                "t": t,
                "x": x,
                **{estimator: est for estimator, est in zip(estimators, x_est)},
            }
        )
        df_k = pd.DataFrame(
            {
                "t": t,
                "x": x,
                **{estimator: est for estimator, est in zip(estimators, std_est)},
            }
        )
        df_x.to_csv(f"./stats/example_x_{component}.csv")
        df_k.to_csv(f"./stats/example_k_{component}.csv")

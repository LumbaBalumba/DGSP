import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dgsp.functions import dim_state
from scripts import ESTIMATORS, NUM_TRAJECTORIES, T_MAX, dt_pred


def example() -> None:
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

    def plot_estimate(i: int) -> None:
        df = pd.read_csv(os.path.join("stats", f"example_x_{i}.csv"))

        plt.figure(figsize=(20, 10))
        plt.plot(df["t"], df["x"], label=True)
        for estimator in ESTIMATORS:
            plt.plot(df["t"], df[estimator], label=estimator.upper())
        plt.legend()

    if not os.path.exists(os.path.join("img", "example")):
        os.makedirs(os.path.join("img", "example"))

    for i in range(dim_state):
        plot_estimate(i)
        plt.savefig(os.path.join("img", "example", f"estimate_{i}.png"))
        plt.clf()

    def plot_err(i: int):
        df_x = pd.read_csv(os.path.join("stats", f"example_x_{i}.csv"))
        df_k = pd.read_csv(os.path.join("stats", f"example_k_{i}.csv"))

        df_x = df_x[len(df_x) // 1000 * 5 :]
        df_k = df_k[len(df_k) // 1000 * 5 :]

        plt.figure(figsize=(20, 10))
        for estimator in ESTIMATORS:
            t = df_x["t"]
            err = df_x[estimator] - df_x["x"]
            sigma = df_k[estimator] ** 0.5

            plt.plot(
                t,
                err,
                label=f"Error {estimator.upper()}",
                color="black",
            )
            plt.plot(
                t,
                sigma,
                label=rf"$\sigma$ {estimator.upper()}",
                color="green",
                marker=".",
            )
            plt.plot(t, -sigma, color="green", marker=".")
            plt.plot(
                t,
                sigma * 3,
                label=rf"$3\sigma$ {estimator.upper()}",
                color="red",
                marker="_",
            )
            plt.plot(t, -sigma * 3, color="red", marker="+")
            plt.legend()
            plt.savefig(os.path.join("img", "example", f"err_{i}_{estimator}.png"))
            plt.clf()

    for i in range(dim_state):
        plot_err(i)


def mass_error() -> None:
    t = np.linspace(0, T_MAX, int(np.ceil(T_MAX / dt_pred)))
    trajs = np.array(
        [
            np.load(os.path.join("data", "traj", f"{i}.npy"))
            for i in range(NUM_TRAJECTORIES)
        ]
    )
    trajs = trajs[:, : len(t), :]
    std = trajs.std(axis=0)

    df = {}

    for estimator in ESTIMATORS:
        trajs_est = np.array(
            [
                np.load(os.path.join("data", "estimate", estimator, "traj", f"{i}.npy"))
                for i in range(NUM_TRAJECTORIES)
            ]
        )
        trajs_est = trajs_est[:, : len(t), :]

        converge = np.array(
            [
                (np.abs(trajs[i] - trajs_est[i]) < std).all()
                for i in range(NUM_TRAJECTORIES)
            ]
        )

        df[estimator] = (trajs_est, converge)

    def error(all: bool) -> None:
        errs = {}

        for estimator in ESTIMATORS:
            if all:
                errs[estimator] = np.std(trajs - df[estimator][0], axis=0)
            else:
                errs[estimator] = np.std(
                    trajs[df[estimator][1], :, :]
                    - df[estimator][0][df[estimator][1], :, :],
                    axis=0,
                )

        for i in range(dim_state):
            plt.figure(figsize=(20, 10))
            for estimator in ESTIMATORS:
                plt.plot(
                    t,
                    errs[estimator][:, i],
                    label=estimator.upper(),
                )
            plt.legend()
            dname = os.path.join("img", "err", "all" if all else "conv")
            if not os.path.exists(dname):
                os.makedirs(dname)
            plt.savefig(os.path.join(dname, f"{i}.png"))

    error(False)
    error(True)


def stats() -> None:
    example()
    mass_error()

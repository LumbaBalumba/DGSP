import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dgsp.functions import dim_state
from scripts import (
    CONVERGENCE_COMPONENTS_CHECK,
    ESTIMATORS,
    NUM_TRAJECTORIES,
    T_MAX,
    dt_pred,
    dt_sim,
)


def example() -> None:
    example_traj_num = 0

    traj = np.load(os.path.join("data", "traj", f"{example_traj_num}.npy"))

    x, y = traj[:, 0], traj[:, 1]
    plt.figure(figsize=(20, 10))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x, y)

    if not os.path.exists("img"):
        os.makedirs("img")

    plt.savefig(os.path.join("img", "traj.png"))
    plt.clf()

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
        std_est = [
            k_est[:, component, component][: len(x)] ** 0.5 for k_est in k_estimates
        ]

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
        plt.xlabel("Время")
        plt.ylabel("Состояние")
        plt.plot(df["t"], df["x"], label="True")
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
            plt.xlabel("Время")
            plt.ylabel("СКО")
            t = df_x["t"]
            err = df_x[estimator] - df_x["x"]
            sigma = df_k[estimator] ** 0.5

            std = np.std(
                np.array(
                    [np.load(f"data/traj/{i}.npy") for i in range(NUM_TRAJECTORIES)]
                ),
                axis=0,
            )
            std = std[:: len(std) // len(t)][: len(t)]

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
    trajs = trajs[:, :: int(np.ceil(dt_pred / dt_sim)), :]
    trajs = trajs[:, : len(t), :]

    df = {}

    for estimator in ESTIMATORS:
        if estimator == "trivial":
            continue
        trajs_est = np.array(
            [
                np.load(os.path.join("data", "estimate", estimator, "traj", f"{i}.npy"))
                for i in range(NUM_TRAJECTORIES)
            ]
        )[:, : len(t), :]

        k_est = np.array(
            [
                np.load(os.path.join("data", "estimate", estimator, "k", f"{i}.npy"))
                for i in range(NUM_TRAJECTORIES)
            ]
        )[:, : len(t), :, :]

        std = np.diagonal(k_est, axis1=2, axis2=3) ** 0.5

        components = CONVERGENCE_COMPONENTS_CHECK
        if components == "all":
            components = np.arange(0, dim_state)
        components = np.array(components)

        converge = np.all(
            np.mean(
                np.abs(trajs[:, :, components] - trajs_est[:, :, components])
                < 5 * std[:, :, components],
                axis=1,
            )
            >= 0.9,
            axis=1,
        )

        df[estimator] = (trajs_est, converge)

    def error(all: bool) -> None:
        errs = {}

        for estimator in ESTIMATORS:
            if estimator == "trivial":
                continue
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
                if estimator == "trivial":
                    continue
                plt.xlabel("Время")
                plt.ylabel("СКО")
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
            plt.clf()

    error(False)
    error(True)

    def diverge_percent() -> None:
        div = [
            (1 - np.mean(df[estimator][1])) * 100
            for estimator in ESTIMATORS
            if estimator != "trivial"
        ]
        df_div = pd.DataFrame(
            {
                "estimator": [
                    estimator.upper()
                    for estimator in ESTIMATORS
                    if estimator != "trivial"
                ],
                "divergence": np.round(div, 2),
            }
        )
        df_div.to_csv(os.path.join("stats", "diverge.csv"), index=False)

    diverge_percent()

    def point_error(idx: int) -> None:
        t_points = np.round(np.linspace(0, T_MAX, int(np.ceil(T_MAX / dt_pred))), 2)
        indices = [len(t_points) // 7 * i for i in range(1, 7, 2)]
        indices = [i in indices for i in range(len(t_points))]
        trajs_idx = trajs[:, :, idx]
        df_err = pd.DataFrame({"t": t_points[indices]})
        for estimator in ESTIMATORS:
            if estimator == "trivial":
                continue
            err = np.mean(np.abs(df[estimator][0][:, :, idx] - trajs_idx), axis=0)
            df_err[estimator] = np.round(err[indices], 5)

        df_err.to_csv(os.path.join("stats", f"error_{idx}.csv"), index=False)

    for i in range(dim_state):
        point_error(i)


def stats() -> None:
    example()
    mass_error()

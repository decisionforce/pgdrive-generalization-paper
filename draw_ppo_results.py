import copy
import json
import numbers
import os
import os.path as osp

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def _flatten_dict(dt, delimiter="/"):
    dt = copy.deepcopy(dt)
    while any(isinstance(v, dict) for v in dt.values()):
        remove = []
        add = {}
        for key, value in dt.items():
            if isinstance(value, dict):
                for subkey, v in value.items():
                    add[delimiter.join([key, subkey])] = v
                remove.append(key)
        dt.update(add)
        for k in remove:
            del dt[k]
    return dt


def _parse(p, contain_config=False):
    dataframe = []
    num_envs = None
    with open(osp.join(p, "result.json"), "r") as f:
        for l in f:
            data = json.loads(l)
            if "environment_num" in data["config"]["env_config"]:
                num_envs = data["config"]["env_config"]["environment_num"]
            else:
                num_envs = 0
            data["num_envs"] = "# envs: {}".format(num_envs)
            data["num_envs_val"] = num_envs
            data["seed"] = data["config"]["seed"]
            if not contain_config:
                data.pop("config")
            data = _flatten_dict(data)
            dataframe.append(data)
    if num_envs is None:
        return None
    dataframe = pd.DataFrame(dataframe)
    return dataframe, num_envs


def parse(root, contain_config):
    """Read and form data into a dataframe

    Usage:

        # Read data
        df = parse(root_path)
        plot_df = smooth(plot_df, 50)
        plot_df = plot_df.sort_values("num_envs_val")

        # Draw figure
        sns.set("notebook", "darkgrid")
        plt.figure(figsize=(8, 6))
        sns.lineplot(
            data=plot_df,
            y="episode_reward_mean",
            x="timesteps_total",
            ci="sd",
            hue="num_envs"
        )
        plt.ylim(-100,)
        plt.title("Train Reward")

    """
    df = []
    paths = [osp.join(root, p) for p in os.listdir(root) if osp.isdir(osp.join(root, p))]
    for pi, p in enumerate(paths):
        print(f"Finish {pi + 1}/{len(paths)} trials.")
        try:
            ret = _parse(p, contain_config)
        except FileNotFoundError:
            print("Path {} not found. Continue.".format(p))
            continue
        if ret is not None:
            df.append(ret)
    if not df:
        print("No Data Found!")
        return None
    df = sorted(df, key=lambda d: d[1])
    df = pd.concat([d for d, c in df])
    return df


def smooth(data, num_points=100, interpolate_x="timesteps_total"):
    data = data.copy()

    if num_points <= 0:
        return data

    trial_list = [j for i, j in data.groupby("experiment_tag")]
    num_points_ = int(max(len(df) for df in trial_list))
    print("Found {} points, draw {} points.".format(num_points_, num_points))
    num_points = min(num_points, num_points_)
    range_min = min(df[interpolate_x].min() for df in trial_list)
    range_max = max(df[interpolate_x].max() for df in trial_list)
    interpolate_range = np.linspace(range_min, range_max, num_points)
    keys = data.keys()
    new_trial_list = []
    for df in trial_list:
        mask = np.logical_and(df[interpolate_x].min() < interpolate_range, interpolate_range < df[interpolate_x].max())
        mask_rang = interpolate_range[mask]
        if len(df) > 1:
            new_df = {}
            df = df.reset_index(drop=True)
            for k in keys:
                if isinstance(df[k][0], numbers.Number):
                    try:
                        new_df[k] = scipy.interpolate.interp1d(df[interpolate_x], df[k])(mask_rang)
                    except ValueError:
                        continue
                elif isinstance(df[k][0], list):
                    continue
                else:
                    new_df[k] = df[k].unique()[0]
            new_trial_list.append(pd.DataFrame(new_df))
        else:
            new_trial_list.append(df)
    return pd.concat(new_trial_list, ignore_index=True)


def filter_nan(df, key="evaluation/episode_reward_mean"):
    return df[~pd.isna(df[key])]


def get_termination(df, evaluate=False, num_points=100):
    """Reorder the dataframe and return a new dataframe for distplot or histplot.

    Usage:

        # Get new df
        new_df = get_termination(df, False)

        # Draw a set of figures
        sns.displot(
            new_df,
            x="timesteps_total",
            hue="done way",
            weights="value",
            multiple="fill",
            element="poly",
            bins=10,
            col="num_envs_val",
        )

        # Draw single figure
        sns.histplot(
            new_df[new_df.num_envs_val==100],
            x="timesteps_total",
            hue="done way",
            weights="value",
            multiple="fill",
            element="poly",
            bins=10,
        )
    """
    dead_keys = {
        "Max Step": "custom_metrics/max_step_rate_mean",
        "Out of Road": "custom_metrics/out_of_road_rate_mean",
        "Crash": "custom_metrics/crash_rate_mean",
        "Success": "custom_metrics/success_rate_mean"
    }
    if evaluate:
        dead_keys = {k: "evaluation/" + v for k, v in dead_keys.items()}
    new_df = []
    df_copy = smooth(df, num_points)
    df_copy = df_copy.sort_values("num_envs_val")
    for name, k in dead_keys.items():
        nd = df_copy.copy()
        nd["done way"] = name
        nd["value"] = nd[k]
        new_df.append(nd)
    new_df = pd.concat(new_df)
    return new_df


if __name__ == '__main__':
    os.makedirs("results", exist_ok=True)
    df = parse("data/main_ppo", True)

    sns.set("talk", "darkgrid")
    colors = sns.color_palette()[:4]
    colors[2], colors[3] = colors[3], colors[2]

    # Draw training result
    new_df = get_termination(df, evaluate=False)
    new_df["num_envs_val"] = new_df["num_envs_val"].astype(int)
    rename_dict = dict(
        #     num_envs="# of Train Env.",
        episode_reward_mean="Episode Reward",
        timesteps_total="Sampled Steps",
        num_envs_val="N",
        **{"done way": "Termination"}
    )
    new_df = new_df.rename(columns=rename_dict)
    ax = sns.displot(
        new_df,
        x=rename_dict["timesteps_total"],
        hue=rename_dict["done way"],
        weights="value",
        multiple="fill",
        element="poly",
        bins=20,
        col=rename_dict["num_envs_val"],
        palette=colors,
        linewidth=.5,
    )
    ax.set_axis_labels("", "Proportion")
    ax.set_xticklabels(step=[], labels=[])
    ax.fig.suptitle("Training Performance", x=0.48, size=32)
    ax.fig.subplots_adjust(top=0.80)
    plt.savefig("results/ppo-main-result-up.pdf", dpi=300, format="pdf", bbox_inches="tight")

    # Draw testing result
    new_df = get_termination(filter_nan(df), evaluate=True, num_points=60)
    new_df["num_envs_val"] = new_df["num_envs_val"].astype(int)
    rename_dict = dict(
        #     num_envs="# of Train Env.",
        episode_reward_mean="Episode Reward",
        timesteps_total="Sampled Steps",
        num_envs_val="N",
        **{"done way": "Termination"}
    )
    new_df = new_df.rename(columns=rename_dict)
    ax = sns.displot(
        new_df,
        x=rename_dict["timesteps_total"],
        hue=rename_dict["done way"],
        weights="value",
        multiple="fill",
        element="poly",
        bins=20,
        col=rename_dict["num_envs_val"],
        palette=colors,
        linewidth=.5,
    )
    ax.set_axis_labels("", "Proportion")
    for i in range(len(ax.axes[0])):
        ax.axes[0][i].ticklabel_format(style='sci', scilimits=(0, 0), axis='x')
    ax.axes[0][3].set_xlabel("Training Steps")

    ax.fig.suptitle("Test Performance", x=0.485, size=32)
    ax.fig.subplots_adjust(top=0.80)
    plt.savefig("results/ppo-main-result-down.pdf", dpi=300, format="pdf", bbox_inches="tight")

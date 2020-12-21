import os

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from draw_ppo_results import parse, filter_nan, get_termination

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
if __name__ == '__main__':
    os.makedirs("results", exist_ok=True)
    df = parse("data/main_sac", True)

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
    plt.savefig("results/sac-main-result-up.pdf", dpi=300, format="pdf", bbox_inches="tight")

    # Draw testing result
    new_df = get_termination(filter_nan(df), evaluate=True)
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
    ax.axes[0][3].set_xlabel("Training Steps")

    ax.fig.suptitle("Test Performance", x=0.485, size=32)
    ax.fig.subplots_adjust(top=0.80)
    plt.savefig("results/sac-main-result-down.pdf", dpi=300, format="pdf", bbox_inches="tight")

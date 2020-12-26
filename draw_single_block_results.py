import os

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from draw_ppo_results import parse, filter_nan, get_termination

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

if __name__ == '__main__':
    os.makedirs("results", exist_ok=True)

    sns.set("talk", "darkgrid")
    colors = sns.color_palette()[:4]
    colors[2], colors[3] = colors[3], colors[2]

    # ===== Draw the single-block agent's results =====
    # Get data
    sbdf = parse("data/single_block_agent", True)
    # Draw training performance
    new_df = get_termination(sbdf, evaluate=False)
    sns.set("paper", "darkgrid")
    plt.figure(figsize=(4, 3))
    ax = sns.histplot(
        new_df,
        x="timesteps_total",
        hue="done way",
        weights="value",
        multiple="fill",
        element="poly",
        bins=50,
        palette=colors,
        linewidth=.5,
        legend=None
    )
    ax.set_ylabel("Proportion")
    ax.set_title("Single-block (Training)", size=16)
    ax.set_xticklabels([])
    ax.set_xlabel("")
    plt.savefig("results/single-block-result-up.pdf", dpi=300, format="pdf", bbox_inches="tight")
    # Draw test performance
    new_df = get_termination(filter_nan(sbdf), evaluate=True, num_points=60)
    sns.set("paper", "darkgrid")
    plt.figure(figsize=(4, 3))
    ax = sns.histplot(
        new_df,
        x="timesteps_total",
        hue="done way",
        weights="value",
        multiple="fill",
        element="poly",
        bins=50,
        palette=colors,
        linewidth=.5,
        legend=None
    )
    ax.set_ylabel("Proportion")
    ax.set_title("Single-block Agent (Test)", size=16)
    ax.set_xlabel("Training Steps")
    plt.savefig("results/single-block-result-down.pdf", dpi=300, format="pdf", bbox_inches="tight")

    # ===== Draw the PG agent's results =====
    # Get data
    pgdf = parse("data/main_ppo", True)
    pgdf = pgdf[pgdf["num_envs_val"] == 100]
    # Draw training performance
    new_df = get_termination(pgdf, evaluate=False)
    new_df["Termination"] = new_df["done way"]
    sns.set("paper", "darkgrid")
    plt.figure(figsize=(4, 3))
    ax = sns.histplot(
        new_df,
        x="timesteps_total",
        hue="Termination",  # So that the legend will show Termination
        weights="value",
        multiple="fill",
        element="poly",
        bins=50,
        palette=colors,
        linewidth=.5,
        legend=None
    )
    ax.set_ylabel("")  # Hide
    ax.set_title("PG Agent (Training)", size=16)
    ax.set_xticklabels([])
    ax.set_xlabel("")
    plt.savefig("results/pg-agent-result-up.pdf", dpi=300, format="pdf", bbox_inches="tight")
    # Draw test performance
    new_df = get_termination(filter_nan(pgdf), evaluate=True, num_points=60)
    sns.set("paper", "darkgrid")
    plt.figure(figsize=(4, 3))
    ax = sns.histplot(
        new_df,
        x="timesteps_total",
        hue="done way",
        weights="value",
        multiple="fill",
        element="poly",
        bins=50,
        palette=colors,
        linewidth=.5,
        # legend=None
    )
    l = ax.legend_
    l._loc = 4
    ax.set_ylabel("")  # Hide
    ax.set_title("PG Agent Agent (Test)", size=16)
    ax.set_xlabel("Training Steps")
    plt.savefig("results/pg-agent-result-down.pdf", dpi=300, format="pdf", bbox_inches="tight")

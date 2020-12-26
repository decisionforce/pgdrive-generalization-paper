import json
import os
import os.path as osp
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pgdrive import GeneralizationRacing
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.execution.rollout_ops import ParallelRollouts
from ray.tune.utils import flatten_dict

from utils import initialize_ray, DrivingCallbacks

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
initialize_ray(test_mode=False)


def get_trainer(friction, checkpoint_path=None, extra_config=None):
    config = dict(
        num_gpus=0,
        num_workers=10,
        num_cpus_per_worker=1,
        horizon=1000,
        lr=0.0,
        batch_mode="complete_episodes",
        callbacks=DrivingCallbacks,

        # Setup the correct environment
        env=GeneralizationRacing,
        env_config=dict(
            # The start seed is default to 0, so the test environments are unseen before.
            environment_num=200,
            vehicle_config=dict(wheel_friction=friction)
        )
    )
    if extra_config:
        config.update(extra_config)
    trainer = PPOTrainer(config=config)
    if checkpoint_path is not None:
        trainer.restore(os.path.expanduser(checkpoint_path))
    return trainer


def evaluate(trainer, num_episodes=20):
    ret_reward = []
    ret_length = []
    ret_success_rate = []
    ret_out_rate = []
    ret_crash_rate = []
    start = time.time()
    episode_count = 0
    while episode_count < num_episodes:
        rollouts = ParallelRollouts(trainer.workers, mode="bulk_sync")
        batch = next(rollouts)
        episodes = batch.split_by_episode()

        ret_reward.extend([e["rewards"].sum() for e in episodes])
        ret_length.extend([e.count for e in episodes])
        ret_success_rate.extend([e["infos"][-1]["arrive_dest"] for e in episodes])
        ret_out_rate.extend([e["infos"][-1]["out_of_road"] for e in episodes])
        ret_crash_rate.extend([e["infos"][-1]["crash"] for e in episodes])

        episode_count += len(episodes)
        print("Finish {} episodes".format(episode_count))

    ret = dict(
        reward=np.mean(ret_reward),
        length=np.mean(ret_length),
        success_rate=np.mean(ret_success_rate),
        out_rate=np.mean(ret_out_rate),
        crash_rate=np.mean(ret_crash_rate)
    )
    print("We collected {} episodes. Spent: {:.3f} s.\nResult: {}".format(
        episode_count, time.time() - start, {k: round(v, 3) for k, v in ret.items()}
    ))
    return ret


def get_result(root, friction_list, num_episodes=20, select_key=None):
    results = []

    root = os.path.abspath(root)
    a_count = 0
    for p in os.listdir(root):
        trial = osp.join(root, p)
        if not osp.isdir(trial):
            continue

        if select_key is not None:
            if select_key not in trial:
                print("We filter out the trial: {} since it does not contain keyword: {}.".format(trial, select_key))
                continue

        assert p.startswith("PPO")
        a_count += 1

        exps = [pp for pp in os.listdir(trial) if pp.startswith("checkpoint")]
        exps.sort(key=lambda v: eval(v.split("_")[1]))
        if not exps:
            print("Empty!")
        assert exps

        # Largest checkpoint index
        ckpt = osp.join(trial, exps[-1], exps[-1].replace("_", "-"))

        config_file = osp.join(trial, "params.json")
        with open(config_file, "r") as f:
            config = json.load(f)
            config = flatten_dict(config)

        for friction in friction_list:
            trainer = get_trainer(friction, ckpt)

            print("\n===== Start Evaluating {} agent =====".format(a_count))
            print("friction: {}\nCheckpoint: {}\n".format(friction, ckpt))

            # evaluate
            ret = evaluate(trainer, num_episodes)

            results.append(dict(path=ckpt, trial=trial, friction=friction, **ret, **config))

            trainer.cleanup()
            del trainer

            print("Finish {} agents.".format(a_count))

    ret = pd.DataFrame(results)
    return ret


if __name__ == '__main__':
    os.makedirs("results", exist_ok=True)
    friction_list = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    # friction_list = [0.8]

    # Get the baseline result
    # friction_result_10 = get_result(
    #     "data/change_friction_10",
    #     friction_list,
    #     100
    # )
    # friction_result_10.to_json("results/change_friction_10.json")
    friction_result_10 = pd.read_json("results/change_friction_10.json")

    # friction_result_06 = get_result(
    #     "data/main_ppo",
    #     friction_list,
    #     100,
    #     "environment_num=100,"
    # )
    # friction_result_06.to_json("results/change_friction_06.json")
    friction_result_06 = pd.read_json("results/change_friction_06.json")

    # Process data
    friction_result_06["Training Friction"] = "Fixed at 0.6"
    friction_result_10["Training Friction"] = "Fixed at 1.0"
    plot_df = pd.concat([friction_result_10.copy(), friction_result_06.copy()])

    # Draw the figure
    sns.set("paper", "darkgrid")
    plt.figure(figsize=(4, 3))
    ax = sns.lineplot(
        data=plot_df,
        x="friction",
        y="success_rate",
        hue="Training Friction",
        ci="sd",
        err_kws=dict(alpha=0.1),
        marker="o",
        ms=7.5
    )
    ax.set_ylabel("Test Success Rate")
    ax.set_xlabel("Test Friction")
    plt.legend(loc="lower right", title="Training Friction")

    # Save the figure
    path = "results/change-friction-result.pdf"
    plt.savefig(path, dpi=300, format="pdf", bbox_inches="tight")
    print("Figure is saved at: ", path)

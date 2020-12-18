from pgdrive import GeneralizationRacing
from ray import tune

from utils import train, get_train_parser

if __name__ == '__main__':
    args = get_train_parser().parse_args()

    exp_name = "main_sac"
    stop = int(1000000)

    config = dict(
        env=GeneralizationRacing,
        env_config=dict(
            environment_num=tune.grid_search([1, 3, 6, 15, 40, 100, 1000]),
            start_seed=tune.grid_search([5000, 6000, 7000, 8000, 9000]),
            low_speed_penalty=0.0,
            steering_penalty=0.1,
            speed_reward=0.1,
        ),

        # ===== Evaluation =====
        evaluation_interval=5,
        evaluation_num_episodes=20,
        evaluation_config=dict(env_config=dict(environment_num=200, start_seed=0)),
        evaluation_num_workers=2,
        metrics_smoothing_episodes=10,

        # ===== Training =====
        optimization=dict(actor_learning_rate=1e-4, critic_learning_rate=1e-4, entropy_learning_rate=1e-4),
        prioritized_replay=True,
        horizon=1000,
        target_network_update_freq=1,
        timesteps_per_iteration=1000,
        learning_starts=10000,
        clip_actions=False,
        normalize_actions=True,
        num_cpus_for_driver=1,
        # No extra worker used for learning. But this config impact the evaluation workers.
        num_cpus_per_worker=0.5,
        # num_gpus_per_worker=0.1 if args.num_gpus != 0 else 0,
        num_gpus=0.5 if args.num_gpus != 0 else 0,
    )

    train(
        "SAC",
        exp_name=exp_name,
        keep_checkpoints_num=5,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        # num_seeds=args.num_seeds,
        num_seeds=1,
        test_mode=args.test,
        # local_mode=True
    )

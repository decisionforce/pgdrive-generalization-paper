from drivingforce.dice.dice_ppo.utils import *
from drivingforce.train import train, get_train_parser
from env_wrapper import StackEnv
from pgdrive import GeneralizationRacing
from ray import tune
from ray.tune import register_env

tf1, tf, tfv = try_import_tf()

if __name__ == '__main__':
    args = get_train_parser().parse_args()


    def _make_env(env_config):
        e = GeneralizationRacing(env_config)
        # TODO get num_stack from env_config
        return StackEnv(e, 3)


    register_env("StackPGDrive", _make_env)
    tmp_env = _make_env(dict())

    exp_name = "context_ppo"
    stop = int(10000000)

    config = dict(
        env="StackPGDrive",
        env_config=dict(
            # environment_num=tune.grid_search([1, 3, 6, 15, 40, 100, 1000]),
            # start_seed=tune.grid_search([5000, 6000, 7000, 8000, 9000]),

            environment_num=tune.grid_search([1, 3, 15, 100]),
            start_seed=tune.grid_search([5000, ]),
        ),

        # ===== Evaluation =====
        evaluation_interval=5,
        evaluation_num_episodes=20,
        evaluation_config=dict(env_config=dict(environment_num=200, start_seed=0)),
        evaluation_num_workers=2,
        metrics_smoothing_episodes=20,

        # ===== Training =====
        horizon=1000,
        num_sgd_iter=20,
        lr=5e-5,
        rollout_fragment_length=200,
        sgd_minibatch_size=100,
        train_batch_size=30000,
        num_gpus=0.75 if args.num_gpus != 0 else 0,
        num_cpus_per_worker=0.2,
        num_cpus_for_driver=1,
        num_workers=5,
    )

    train(
        "PPO",
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

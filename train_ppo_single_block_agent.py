from pgdrive import GeneralizationRacing
from pgdrive.scene_creator.algorithm.BIG import BigGenerateMethod
from pgdrive.scene_creator.map import Map
from ray import tune

from utils import train, get_train_parser

if __name__ == '__main__':
    args = get_train_parser().parse_args()

    exp_name = "single_block_agent"
    stop = int(10000000)

    config = dict(
        env=GeneralizationRacing,
        env_config=dict(
            environment_num=tune.grid_search([300]),
            start_seed=tune.grid_search([5000, 6000, 7000, 8000, 9000]),
            map_config={
                Map.GENERATE_METHOD: BigGenerateMethod.BLOCK_NUM,
                Map.GENERATE_PARA: 1
            },
        ),

        # ===== Evaluation =====
        evaluation_interval=5,
        evaluation_num_episodes=20,
        evaluation_config=dict(env_config=dict(environment_num=200, start_seed=0, map_config={
            Map.GENERATE_METHOD: BigGenerateMethod.BLOCK_NUM,
            Map.GENERATE_PARA: 3
        }, )),
        evaluation_num_workers=2,
        metrics_smoothing_episodes=20,

        # ===== Training =====
        horizon=1000,
        num_sgd_iter=20,
        lr=5e-5,
        rollout_fragment_length=200,
        sgd_minibatch_size=100,
        train_batch_size=30000,
        num_gpus=0.5 if args.num_gpus != 0 else 0,
        num_cpus_per_worker=0.25,
        num_cpus_for_driver=1,
        num_workers=10,
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

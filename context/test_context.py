import pgdrive
from context_model import FullyConnectedNetworkWithContext
from drivingforce.dice.dice_ppo.utils import *
from drivingforce.train import initialize_ray
from env_wrapper import StackEnv
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.models import ModelCatalog
from ray.tune import register_env

tf1, tf, tfv = try_import_tf()


def make_model(policy, obs_space, action_space, config, name=None):
    dist_class, logit_dim = ModelCatalog.get_action_dist(action_space, config["model"])
    return FullyConnectedNetworkWithContext(obs_space, action_space, logit_dim, model_config=config["model"], name=name)


ContextPPOTFPolicy = PPOTFPolicy.with_updates(
    name="ContextPPOTFPolicy",
    make_model=make_model
)

ContextPPO = PPOTrainer.with_updates(
    name="ContextPPO",
    default_policy=ContextPPOTFPolicy,
    get_policy_class=lambda *_: ContextPPOTFPolicy,
)

if __name__ == '__main__':
    initialize_ray(test_mode=True, local_mode=False)


    def _make_env(env_config):
        e = pgdrive.PGDriveEnv()
        # TODO get num_stack from env_config
        return StackEnv(e, 3)


    register_env("TestEnv", _make_env)
    tmp_env = _make_env(dict())

    t = ContextPPO(
        config=dict(num_workers=0, env="TestEnv"),
        # obs_space=tmp_env.observation_space,
        # action_space=tmp_env.action_space
    )

    for _ in range(5):
        print(t.train())

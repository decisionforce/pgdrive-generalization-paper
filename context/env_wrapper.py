from collections import deque, OrderedDict

import gym
import numpy as np
from gym.spaces import Box, Tuple


class StackEnv(gym.Wrapper):

    def __init__(self, env, num_stacks, return_latest_obs=True, *args, **kwargs):
        super(StackEnv, self).__init__(env)

        assert isinstance(num_stacks, int)
        assert num_stacks > 0
        self.num_stacks = num_stacks
        self.return_latest_obs = return_latest_obs
        assert self.return_latest_obs, "Force this option all the time."

        assert isinstance(self.action_space, Box)
        assert len(self.action_space.shape) == 1, "Only support continuous action space now!"
        self.act_dim = self.action_space.shape[0]
        self.stacks = OrderedDict()
        self._prev_obs = None

        self.low = self.action_space.low[0] if np.isfinite(self.action_space.low[0]) else -1.0
        self.high = self.action_space.high[0] if np.isfinite(self.action_space.high[0]) else 1.0
        self.obs_low = self.observation_space.low[0] if np.isfinite(self.observation_space.low[0]) else -1.0
        self.obs_high = self.observation_space.high[0] if np.isfinite(self.observation_space.high[0]) else 1.0

        if isinstance(self.observation_space, Box):
            shape = self.observation_space.shape
            assert len(shape) == 1, "Only support scalar observation now!"
            self.original_obs_dim = shape[0]
            self.obs_dim = (shape[0] + self.act_dim + 1) * self.num_stacks
            if self.return_latest_obs:
                self.obs_dim += self.original_obs_dim
                space = Tuple([
                    Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim - self.original_obs_dim,)),
                    Box(low=-np.inf, high=+np.inf, shape=(self.original_obs_dim,)),
                ])
            else:
                raise ValueError()
            self.observation_space = space

        else:
            raise NotImplementedError("Only support Box space now!")

    def step(self, action):
        obs, reward, done, info = super(StackEnv, self).step(action)
        new_obs = self.stack(action, self._prev_obs, reward, latest_obs=obs)
        self._prev_obs = np.asarray(obs).copy()
        return new_obs, reward, done, info

    def stack(self, action, obs, reward, latest_obs):
        self.stacks['action'].append(self._normalize_action(action))
        self.stacks['obs'].append(obs)
        self.stacks['reward'].append(self._clip_reward(reward))
        return self.get_obs(latest_obs)

    def get_obs(self, latest_obs):
        if self.return_latest_obs:
            ret = np.concatenate(
                [np.array(stack, dtype=np.float).reshape(-1) for stack in self.stacks.values()]
            )
            ret = [ret, latest_obs]
        else:
            raise ValueError("Not supported yet!")
        assert len(ret[0]) + len(ret[1]) == self.obs_dim
        return ret

    def _normalize_action(self, action):
        new_action = (action - self.low) / (self.high - self.low) * (self.obs_high - self.obs_low) + self.obs_low
        return new_action

    def _clip_reward(self, reward):
        # Don't clip the reward here. I am afraid different environment might have different reward range.
        # return max(min(reward, self.obs_high), self.obs_low)
        return reward

    def init_stack(self):
        num_stacks = self.num_stacks
        action_stack = deque(
            [np.zeros(shape=self.act_dim, dtype=np.float) for _ in range(num_stacks)], maxlen=num_stacks
        )
        obs_stack = deque(
            [np.zeros(shape=self.original_obs_dim, dtype=np.float) for _ in range(num_stacks)], maxlen=num_stacks
        )
        reward_stack = deque(
            [np.zeros(shape=1, dtype=np.float) for _ in range(num_stacks)], maxlen=num_stacks
        )
        self.stacks = OrderedDict(
            obs=obs_stack,
            action=action_stack,
            reward=reward_stack
        )

    def reset(self, **kwargs):
        obs = super(StackEnv, self).reset(**kwargs)
        self._prev_obs = obs
        self.init_stack()
        return self.get_obs(obs)


if __name__ == '__main__':
    env = gym.make("BipedalWalker-v3")
    env = StackEnv(env, num_stacks=3)
    env.reset()
    for _ in range(10):
        o, r, d, i = env.step(env.action_space.sample())
        print(o, r, d, i)
    env.close()

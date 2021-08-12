from algos.sac import SACModel, SACAGent_DISC
from algos.utils import CustomEnvWrapper

from easydict import EasyDict

from pommerman.configs import one_vs_one_env, team_competition_fast_env
import pommerman.envs as envs
from pommerman import constants, characters
from pommerman.agents import SimpleAgent
from pommerman.characters import Bomber

import gym
import pommerman
import numpy as np

from pommerman.envs.v0 import Pomme

from rl2.workers.base import RolloutWorker
from rl2.models.torch.base import BaseEncoder
from rl2.buffers.prioritized import PrioritizedReplayBuffer

import pprint

env_config = one_vs_one_env()
env_config['env_kwargs']['agent_view_size'] = 4
env = CustomEnvWrapper(env_config)

myconfig = {
    'gamma': 0.99,
    'train_interval': 1,
    'update_interval': 1,
    'batch_size': 128
}

config = EasyDict(myconfig)

action_shape = env.action_space.n if hasattr(env.action_space,
                                             'n') else env.action_space.shape

observation_shape, additional_shape = env.observation_shape

model = SACModel(observation_shape,
                 action_shape,
                 discrete=True,
                 injection_shape=additional_shape)
# observation: tuple, action_shape: int

trainee_agent = SACAGent_DISC(model,
                              batch_size=config.batch_size,
                              train_interval=config.train_interval,
                              update_interval=config.update_interval,
                              save_interval=1000,
                              character=Bomber(0, env_config["game_type"]))

agents = {
    0: trainee_agent,
    1: SimpleAgent(env_config['agent'](1, env_config["game_type"])),
}

env.set_trainee_agents(0)
#print(env._trainee_agents)
env.set_agents(list(agents.values()))
env.set_init_game_state(None)
env.seed(44)

obs = env.reset()
print(len(obs))


class SimpleWorker:
    def __init__(self, env, trainee_agent, training, render, max_step_size,
                 is_save, **kwargs):
        self.env = env
        self.agent = trainee_agent
        self.training = training
        self.render = render
        self.is_save = is_save
        self.max_step_size = max_step_size

    def rollout(self):
        self.agent.curr_step = 0
        obs = self.env.reset()
        done = False
        while not done and self.agent.curr_step < self.max_step_size:
            actions = self.env.act(obs)
            print(actions)
            obs_, rewards, done, info = self.env.step(actions)
            self.agent.step(obs[0], actions[0], rewards[0], done, obs_[0])
            obs = obs_


# get partial obs

worker = SimpleWorker(env,
                      trainee_agent,
                      training=True,
                      render=True,
                      max_step_size=1000,
                      is_save=True)

worker.rollout()
#print(obs)
#for i, p in enumerate(obs):
#    print(i)
#    for k, v in p.items():
#        print(k, v)

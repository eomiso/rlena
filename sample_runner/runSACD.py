from algos.sac_discrete import SACAgentDISC, SACModelDISC

from easydict import EasyDict

from pommerman.configs import one_vs_one_env, team_competition_fast_env
import pommerman.envs as envs
from pommerman import constants, characters
from pommerman.agents import SimpleAgent
from pommerman.characters import Bomber
from utils.wrapper import ConservativeEnvWrapper

import gym
import pommerman
import numpy as np

from pommerman.envs.v0 import Pomme

from rl2.workers.base import RolloutWorker
from rl2.models.torch.base import BaseEncoder
from rl2.buffers.base import ReplayBuffer
from utils.worker import SimpleWorker

import pprint

env_config = one_vs_one_env()
env_config['env_kwargs']['agent_view_size'] = 4
env_config['env_kwargs']['max_step'] = 200
env = ConservativeEnvWrapper(env_config)

myconfig = {
    'gamma': 0.99,
    'train_interval': 5,
    'train_after': 1024,
    'update_interval': 30,
    'update_after': 1024,
    'save_interval': 1000,
    'batch_size': 128,
    'random_until': 0,
    'render': True,
}

config = EasyDict(myconfig)

action_shape = env.action_space.n if hasattr(env.action_space,
                                             'n') else env.action_space.shape

observation_shape, additional_shape = env.observation_shape


def obs_handler(obs, keys=['locational', 'additional']):
    if isinstance(obs, dict):
        loc, add = [obs[key] for key in keys]
    else:
        loc = []
        add = []
        for o in obs:
            loc.append(o[0]['locational'])
            add.append(o[0]['additional'])
        loc = np.stack(loc, axis=0)
        add = np.stack(add, axis=0)
    return loc, add


model = SACModelDISC(observation_shape, (action_shape, ),
                     discrete=True,
                     injection_shape=additional_shape,
                     preprocessor=obs_handler,
                     is_save=True)

load_dir = 'sac_discrete/test/ckpt/2021Aug19_20_06_53/1k'
model.load(load_dir)
# observation: tuple, action_shape: int

buffer_kwargs = {
    'size': 1e6,
    'elements': {
        'obs': ((5, 9, 9), (8, ), np.float32),
        'action': ((6, ), np.float32),
        'reward': ((1, ), np.float32),
        'done': ((1, ), np.float32),
        'obs_': ((5, 9, 9), (8, ), np.float32)
    }
}

trainee_agent = SACAgentDISC(model,
                             batch_size=config.batch_size,
                             train_interval=config.train_interval,
                             update_interval=config.update_interval,
                             save_interval=100,
                             buffer_cls=ReplayBuffer,
                             buffer_kwargs=buffer_kwargs,
                             save_dir='sac_discrete/runner',
                             log_dir='sac_discrete/runner',
                             character=Bomber(0, env_config["game_type"]))

agents = {
    0: trainee_agent,
    1: SimpleAgent(env_config['agent'](1, env_config["game_type"])),
}

env.set_init_game_state(None)
env.set_agents(list(agents.values()))
env.set_training_agents(0)
env.seed(44)

worker = SimpleWorker(env,
                      agents=[trainee_agent],
                      render=config.render,
                      max_step=200,
                      random_until=config.random_until,
                      training=False)

print(worker.__dict__)

worker.run()

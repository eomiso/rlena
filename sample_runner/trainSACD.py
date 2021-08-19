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

import torch

import pprint

import argparse

argparser = argparse.ArgumentParser(description='SACD Single agent')
argparser.add_argument('--cuda_device', type=int, default=0)
argparser.add_argument('--train_interval', type=int, default=1)
argparser.add_argument('--update_interval', type=int, default=5)
argparser.add_argument('--max_step', type=int, default=1000)

args = argparser.parse_args()

args.device = torch.device(
    'cuda:{}'.format(args.cuda_device) if torch.cuda.is_available() else 'cpu')

env_config = one_vs_one_env()
env_config['env_kwargs']['agent_view_size'] = 4
env_config['env_kwargs']['max_step'] = args.max_step
env = ConservativeEnvWrapper(env_config)

myconfig = {
    'gamma': 0.99,
    'train_interval': args.train_interval,
    'train_after': 1024,
    'update_interval': 30,
    'update_after': 1024,
    'save_interval': 1000,
    'batch_size': 128,
    'max_step': args.max_step,
    'random_until': 1024,
    'device': args.device
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
                     is_save=True,
                     device=config.device)
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
                             train_after=config.train_after,
                             update_interval=config.update_interval,
                             update_after=config.update_after,
                             save_interval=config.save_interval,
                             buffer_cls=ReplayBuffer,
                             buffer_kwargs=buffer_kwargs,
                             save_dir='sac_discrete/train/ckpt',
                             log_dir='sac_discrete/train/log',
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
                      render=False,
                      max_step=config.max_step,
                      random_until=config.random_until)

print(worker.__dict__)

worker.run()

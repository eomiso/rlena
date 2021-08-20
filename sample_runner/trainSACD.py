from easydict import EasyDict
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path

from algos.sac_discrete import SACAgentDISC, SACModelDISC
from utils.wrapper import ConservativeEnvWrapper

from pommerman.configs import one_vs_one_env, team_competition_fast_env
import pommerman.envs as envs
from pommerman.agents import SimpleAgent
from pommerman.characters import Bomber

from rl2.buffers.base import ReplayBuffer
from rl2.examples.temp_logger import Logger
from utils.worker import SACDworker

import torch

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


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='SACD Single agent')
    argparser.add_argument('--cuda_device', type=int, default=0)
    argparser.add_argument('--train_interval', type=int, default=1)
    argparser.add_argument('--update_interval', type=int, default=5)
    argparser.add_argument('--max_step', type=int, default=1000)
    argparser.add_argument('--dir_name', type=str, default='')
    args = argparser.parse_args()

    args.device = 'cuda:{}'.format(args.cuda_device) if torch.cuda.is_available() else 'cpu'

    time = datetime.now().strftime('%Y%b%d_%H_%M_%S')
    config = EasyDict({
        'gamma': 0.99,
        'n_agents': 2,
        'n_env': 1,
        'train_interval': args.train_interval,
        'train_after': 5098,
        'update_interval': args.update_interval,
        'update_after': 5098,
        'rand_until': 5098,
        'save_interval': 1000,
        'batch_size': 64,
        'max_step': args.max_step,
        'random_until': 5098,
        'device': args.device,
        'render':True,
        'render_interval': 100,
        'log_interval': 10, 
        'eps': 0.5, # for decaying epsilon greedy
        'log_dir': f'./sac-discrete/train/log/{time}',
        'save_dir': f'./sac-discrete/train/ckpt/{time}'
    })

    env_config = one_vs_one_env()
    env_config['env_kwargs']['agent_view_size'] = 4
    env_config['env_kwargs']['max_step'] = args.max_step
    env = ConservativeEnvWrapper(env_config)

    action_shape = env.action_space.n if hasattr(env.action_space,
                                                'n') else env.action_space.shape
    observation_shape, additional_shape = env.observation_shape


    model = SACModelDISC(observation_shape, (action_shape, ),
                        discrete=True,
                        injection_shape=additional_shape,
                        preprocessor=obs_handler,
                        is_save=True,
                        device=config.device)

    # observation: tuple, action_shape: int
    trainee_agent = SACAgentDISC(model,
                                batch_size=config.batch_size,
                                train_interval=config.train_interval,
                                train_after=config.train_after,
                                update_interval=config.update_interval,
                                update_after=config.update_after,
                                render_interval=config.render_interval,
                                save_interval=config.save_interval,
                                buffer_cls=ReplayBuffer,
                                buffer_kwargs=buffer_kwargs,
                                save_dir=config.save_dir,
                                eps = config.eps,
                                rand_until=config.rand_util,
                                # log_dir='sac_discrete/train/log',
                                character=Bomber(0, env_config["game_type"]))

    agents = {
        0: trainee_agent,
        1: SimpleAgent(env_config['agent'](1, env_config["game_type"])),
    }

    env.set_init_game_state(None)
    env.set_agents(list(agents.values()))
    env.set_training_agents(0)
    env.seed(44)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    logger = Logger(name='TestSACD', args=config)
    worker = SACDworker(env,
                        agents=[trainee_agent],
                        n_agents=config.n_agents,
                        n_env=config.n_env,
                        max_episodes=3e4,
                        training=True,
                        logger=logger, 
                        log_interval=config.log_interval,
                        render=config.render,
                        render_interval=config.render_interval,
                        is_save= True,
                        random_until=config.random_until)

    worker.run()

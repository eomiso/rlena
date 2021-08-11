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

env_config = one_vs_one_env()
env = Pomme(**env_config['env_kwargs'])

myconfig = {
    'gamma': 0.99,
    'train_interval': 1,
    'update_interval': 1,
    'batch_size': 128
}

config = EasyDict(myconfig)

action_shape = env.action_space.n if hasattr(env.action_space,
                                             'n') else env.action_space.shape
model = SACModel(env.observation_space.shape, action_shape, discrete=True)
# observation: tuple, action_shape: int

trainee_agent = SACAGent_DISC(model,
                              batch_size=config.batch_size,
                              train_interval=config.train_interval,
                              update_interval=config.update_interval,
                              character=Bomber(0, env_config["game_type"]))

agents = {
    0: trainee_agent,
    1: SimpleAgent(env_config['agent'](1, env_config["game_type"])),
}

env.set_agents(list(agents.values()))
env.set_init_game_state(None)
env.seed(44)


class SimpleMaxStepWorker(RolloutWorker):
    """
    do rollout until max steps given
    """
    def __init__(self,
                 env,
                 n_env,
                 agent,
                 max_steps=1000,
                 logger=None,
                 log_interval=5000,
                 **kwargs):
        super().__init__(env, n_env, agent, **kwargs)
        self.max_steps = int(max_steps)
        self.log_interval = int(log_interval)
        self.logger = logger
        self.info = {}

    def run(self):
        for step in range(self.max_steps // self.n_env + 1):
            done, info, results = self.rollout()

            # Log info
            if self.num_steps % self.log_interval < self.n_env:
                info_r = {
                    'Counts/num_steps': self.num_steps,
                    'Counts/num_episodes': self.num_episodes,
                    'Episodic/rews_avg': np.mean(list(self.scores)),
                    'Episodic/ep_length': np.mean(list(self.ep_length))
                }
                self.info.update(info_r)
                # self.info.update(info)
                self.logger.scalar_summary(self.info, self.num_steps)

            # Save model
            if (self.is_save
                    and self.num_steps % self.save_interval < self.n_env):
                if hasattr(self, 'logger'):
                    save_dir = getattr(self.logger, 'log_dir')
                else:
                    save_dir = os.getcwd()
                self.save(save_dir)


# get partial obs
print(env.observation_space)
print(env.action_space)

worker = SimpleMaxStepWorker(env, 1, trainee_agent)

worker.run()
#print(obs)
#for i, p in enumerate(obs):
#    print(i)
#    for k, v in p.items():
#        print(k, v)

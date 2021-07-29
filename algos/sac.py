from easydict import EasyDict
from typing import Callable, List

import numpy as np
import os

import torch
import torch.nn.functional as F

import gym
import pommerman
from pommerman.configs import one_vs_one_env, team_competivition_env


from rl2.models.torch.actor_critic import ActorCriticModel
from rl2.agents.base import MAgent, Agent   
from rl2.buffers import ReplayBuffer

def loss_func_ac(data,model, **kwargs):
    s, a, r, d, s_, adv = tuple(
        map(lambda x: torch.from_numpy(x).float().to(model.device), data)
    )
    ac_dist, val_dist = model(s, hidden=kwargs.hidden, mask=d)
    #TODO
    return loss

def loss_func_cr(data, model, **kwargs):
    #TODO
    pass


def loss_func(data, model, **kwargs): 
    actor_loss = loss_func_ac(data, model)
    critic_loss = loss_func_cr(data,model)

    return actor_loss, critic_loss

class SACModel(ActorCriticModel):
    pass


class SACAGent(Agent):
    def __init__(self,
                 model=SACModel,
                 n_env=1,
                 buffer_cls: ReplayBuffer=ReplayBuffer,
                 buffer_kwargs: dict=None, #TODO
                 batch_size: int=None, #TODO
                 train_interval: int=None, #TODO
                 num_epochs: int=1, #TODO
                 loss_func: Callable=loss_func, 
                 **kwargs):

        if 'config' in kwargs.keys():
            self.config = EasyDict(kwargs.get('config'))
        self.init_collect = self.config.init_collect
        self.train_interval = self.config.train_interval
        self.batch_size = self.config.batch_size
        self.gamma = self.config.gamma

        if buffer_kwargs is None:
            buffer_kwargs = {'size': self.config.buffer}
        super().__init__(model, train_interval, num_epochs, buffer_cls, buffer_kwargs)
    
    def train(self, batch):
        pass

    def act(self, obs):
        pass

    def collect(self):
        pass

class SACMAGENT(MAgent):
    def __init__(self):
        #TODO
        pass

    def train(self, **kwargs):
        #TODO
        pass

    def act(self, **kwargs):
        #TODO
        pass

    def train(self):
        #TODO
        pass


# Try out the agent
if __name__ == '__main__':
    pass
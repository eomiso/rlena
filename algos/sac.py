from easydict import EasyDict
from typing import Callable, List

import numpy as np
import os

import torch
import torch.nn.functional as F

import gym
import pommerman
from pommerman.configs import one_vs_one_env, team_competivition_env


from rl2.models.torch.actor_critic import TorchModel
from rl2.agents.base import MAgent, Agent   
from rl2.buffers import ReplayBuffer

def loss_func_ac(data, model, **kwargs):
    s, a, r, d, s_ = tuple(
        map(lambda x: torch.from_numpy(x).float().to(model.device), data)
        )
    ac_dist = model.infer(s, hidden=kwargs.hidden, mask=d)

    if model.discrete:
        pass
    else:
        pass


def loss_func_cr(data, model, **kwargs):
    s, a , r, d, s_ = tuple(
        map(lambda x: torch.from_numpy(x).float().to(model.device), data)
        )
    with torch.no_grad():
        if model.discrete:

            pass
        else:
            pass

def loss_func_alph(data, model, **kwargs):
    pass


class SACModel(TorchModel):
    def __init__(self, 
                 obs_shape, 
                 action_shape,
                 actor: torch.nn.Module = None, 
                 critic: torch.nn.Module = None, 
                 encoder: torch.nn.Module = None, 
                 encoded_dim: int = 64,
                 optim_ac: str = 'torch.nn.Module',
                 optim_cr: str = 'torch.nn.Module',
                 lr_ac: float = 1e-4,
                 lr_cr: float = 1e-4,
                 grad_clip: float = 1e-2,
                 polyak: float = 0.995,
                 discrete: bool = False,
                 flatten: bool = False,
                 reorder: bool = False,
                 recurrent: bool = False,
                 **kwargs):
        super().__init__(obs_shape, action_shape, **kwargs)
        if hasattr(encoder, 'output_shape'):
            encoded_dim = encoder.output_shape
        self.encoded_dim = encoded_dim
        self.recurrent = recurrent
        self.discrete = discrete
        self.flatten = flatten
        self.reorder = reorder

        self.optim_ac = optim_ac
        self.optim_cr = optim_cr
        
        self.lr_ac = lr_ac
        self.lr_cr = lr_cr
        
        self.grad_clip = grad_clip
        self.polyak = polyak

        self.is_save = kwargs.get('is_save', False)

        # stochastic policy network
        self.pi = 

        # q network 
        self.q1 = 
        self.q2 = 
        
        self.init_param(self.pi)
        self.init_param(self.q1)
        self.init_param(self.q2)


    #@TorchModel.sharedbranch
    def forward(self, obs, **kwargs):
        obs = obs.to(self.device)

        
    def infer(self, x):
        ir = self.encoder(x)
        ac_dist = self.actor(ir)

        return ac_dist

    def step(self, loss):
        self.optimizer.zero_grad()

        loss.backward()

        nn.utils.clip_grad_norm(self.actor.parameters(), self.max_grad)
        nn.utils.clip_grad_norm(self.encoder.parameters(), self.max_grad)
        nn.utils.clip_grad_norm(self.critic.parameters(), self.max_grad)

        self.optimizer.step()

    def save(self, save_dir):
        # torch.save(os.path.join(save_path, 'actor_critic.pt'), self.state_dict())
        torch.save(os.path.join(save_dir, 'encoder.pt'))
        torch.save(os.path.join(save_dir, 'actor.pt'))
        torch.save(os.path.join(save_dir, 'critic.pt'))

    def load(self, save_dir):
        self.encoder = torch.load(os.path.join(save_dir, 'encoder.pt'))
        self.actor = torch.load(os.path.join(save_dir, 'actor.pt'))
        self.critic = torch.load(os.path.join(save_dir, 'critic.pt'))



class SACAGent_DISC(Agent):
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



class SACMAGENT_DISC(MAgent):
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
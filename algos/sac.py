from easydict import EasyDict
from typing import Callable, List, Iterable

import numpy as np
import os

import torch
import torch.nn.functional as F

import gym
import pommerman
from pommerman.agents import BaseAgent
from pommerman.configs import one_vs_one_env, team_competition_env

import torch as T
import torch.nn as nn
from rl2.models.torch.base import InjectiveBranchModel, TorchModel, BranchModel
from rl2.agents.base import MAgent, Agent
from rl2.buffers.base import ReplayBuffer
from rl2.buffers.prioritized import PrioritizedReplayBuffer
from algos.buffer import CustomBuffer

import pprint


def loss_func_ac(data, model, **kwargs):
    weights = kwargs['weights']

    obs, a, r, done, obs_ = tuple(
        map(lambda x: T.from_numpy(x).float().to(model.device), datas))

    if model.encoder is not None:
        states = model.encoder(obs)
    else:
        states = obs

    act_probs, log_act_probs, _ = model.pi(states)
    with T.no_grad():
        q1 = model.q1.forward(state)
        q2 = model.q2.forward(state)
        q = T.min(q1, q2)
    # (Log of) probabilities to calculate expectations of Q and entropies.
    # Expectations of q
    q = T.sum(q * act_probs, dim=1, keepdim=True)
    # Expectations of entropies.
    entropies = -T.sum(act_probs * log_act_probs, dim=1, keepdim=True)
    # Policy objective is maximization of (Q+alpha*entropy) with
    # priority weights
    l_ac = T.mean(weights * (-q - model.alpha * entropies))

    return l_ac, entropies.detach()


def loss_func_cr(data, model, **kwargs):
    weights = kwargs['weights']

    import pdb
    pdb.set_trace()
    obs, a, r, d, obs_ = tuple(
        map(lambda x: torch.from_numpy(x).float().to(model.device), data))

    if model.encoder is not None:
        states = model.encoder(obs)
    else:
        states = obs

    curr_q1 = model.q1.forward(states)
    curr_q2 = model.q2.forward(states)
    target_q1 = model.q1.forward_trg(states)
    target_q2 = model.q2.forward_trg(states)

    # TD errors for updating priority weights
    errors = T.abs(curr_q1.detach() - target_q)

    # log means of Q to monitor training
    mean_q1 = curr_q1.detach().mean().item()
    mean_q2 = curr_q2.detach().mean().item()

    # critic loss is the mean squared TD errors with priority weights
    l_cr1 = T.mean((curr_q1 - target_q1).pow(2) * weights)
    l_cr2 = T.mean((curr_q2 - target_q2).pow(2) * weights)

    return l_cr1, l_cr2, errors, mean_q1, mean_q2


def loss_func_alpha(entropies, model, **kwargs):
    weights = kwargs['weights']

    # Intuitively, we increse alpha when entropy is less than target
    # entropy, vice versa.
    l_entropy = -torch.mean(model.log_alpha *
                            (model.target_entropy - entropies) * weights)

    return l_entropy


class SACModel(TorchModel):
    def __init__(
            self,
            obs_shape,
            action_shape,
            actor: torch.nn.Module = None,
            critic: torch.nn.Module = None,
            encoder: torch.nn.Module = None,
            encoded_dim: int = 64,
            optim_ac: str = 'torch.optim.Adam',
            optim_cr: str = 'torch.optim.Adam',
            lr_ac: float = 1e-4,
            lr_cr: float = 1e-4,
            grad_clip: float = 1e-2,
            polyak: float = 0.995,
            discrete: bool = False,
            flatten: bool = False,  # True if you don't need CNN in the encoder
            reorder: bool = False,  # Flag for (C, H, W)
            additional: bool = False,
            deterministic: bool = False,
            device: str = None,
            **kwargs):

        self.device = device  # TODO
        assert 'injection_shape' in kwargs, "You need injection_shape for additional info"
        self.injection_shape = kwargs['injection_shape']

        super().__init__(obs_shape, (action_shape, ), **kwargs)
        if hasattr(encoder, 'output_shape'):
            encoded_dim = encoder.output_shape
        self.encoer = encoder
        self.encoded_dim = encoded_dim
        self.discrete = discrete
        self.flatten = flatten
        self.reorder = reorder

        self.lr_ac = lr_ac
        self.lr_cr = lr_cr

        self.grad_clip = grad_clip
        self.polyak = polyak

        self.is_save = kwargs.get('is_save', False)

        self.eps = np.finfo(np.float32).eps.item()

        self.use_automatic_entropy_tuning = kwargs.get(
            'use_automatic_entropy_tuning', True)
        if self.use_automatic_entropy_tuning:
            # set the max possible entropy as the target entropy
            if (isinstance(self.action_shape, Iterable)):
                self.target_entropy = -np.log(1 / self.action_shape[0]) * .98
            else:
                self.target_entropy = -np.log(1 / self.action_shape) * .98
            self.target_entropy = T.tensor(self.target_entropy).to(self.device)
            self.log_alpha = T.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = T.optim.Adam([self.log_alpha],
                                            lr=lr_ac,
                                            eps=self.eps)

        # stochastic policy network
        self.pi = InjectiveBranchModel(
            obs_shape,
            (action_shape, ),
            #injection_shape=additional_shape,
            encoder=encoder,
            optimizer=optim_ac,
            lr=lr_ac,
            discrete=discrete,
            deterministic=deterministic,
            reorder=reorder,
            flatten=flatten,
            **kwargs)

        # q network
        self.q1 = InjectiveBranchModel(
            obs_shape,
            (action_shape, ),
            #injection_shape=additional_shape,
            encoder=encoder,
            encoded_dim=encoded_dim,
            optimizer=optim_cr,
            lr=lr_cr,
            discrete=discrete,
            deterministic=deterministic,
            flatten=flatten,
            reorder=reorder,
            make_target=True,
            **kwargs)

        self.q2 = InjectiveBranchModel(
            obs_shape,
            (action_shape, ),
            #injection_shape=injection_shape,
            encoder=encoder,
            encoded_dim=encoded_dim,
            optimizer=optim_cr,
            lr=lr_cr,
            discrete=discrete,
            deterministic=deterministic,
            flatten=flatten,
            reorder=reorder,
            make_target=True,
            **kwargs)

        self.networks = nn.ModuleDict({
            'policy': self.pi,
            'q1': self.q1,
            'q2': self.q2,
        })

        self.init_params(self.pi)
        self.init_params(self.q1)
        self.init_params(self.q2)

    @TorchModel.sharedbranch
    def forward(self, obs, **kwargs):
        #pprint.pprint(obs[0])
        #obs = T.tensor(obs)
        #obs = obs.to(self.device)
        #state = self.encoder(obs)
        locational = obs.get('locational')
        locational = T.tensor(locational, dtype=T.float32)
        injection = obs.get('additional')
        injection = T.tensor(injection, dtype=T.float32)
        act_probs = self.pi(locational, injection)
        # deal with log nan with log 0s
        act_log_probs = T.log(act_probs.probs.detach()[0],
                              out=T.tensor(self.eps))
        max_act = T.argmax(act_probs.probs)

        return act_probs, act_log_probs, max_act

    def act(self, obs):
        act_probs, _, _ = self.forward(obs)
        action = act_probs.sample()

        return action

    def update_trg(self):
        # soft target update
        self.q1.update_trg()
        self.q2.update_trg()

    def save(self, save_dir):
        T.save(self.networks, os.path.join(save_dir, 'sac_discrete.pt'))
        print('model saved in {}'.format(save_dir))

    def load(self, save_dir):
        self.networks = T.load(os.path.join(save_dir, 'sac_discrete.pt'),
                               map_location=self.device)
        self.pi.load_state_dict(self.networks['policy'])
        self.q1.load_state_dict(self.networks['q1'])
        self.q2.load_state.dict(self.networks['q2'])
        print('model loaded from {}'.format(save_dir))


class SACAGent_DISC(Agent, BaseAgent):
    def __init__(self,
                 model: TorchModel,
                 buffer_cls: ReplayBuffer = ReplayBuffer,
                 buffer_size: int = int(1e6),
                 buffer_kwargs: dict = None,
                 batch_size: int = None,
                 num_epochs: int = None,
                 update_interval: int = 5,
                 train_interval: int = 10,
                 train_after: int = 10,
                 update_after: int = 5,
                 use_per: bool = False,
                 **kwargs):

        # for pommerman need to initialiize the base agent
        if 'character' in kwargs:
            self._character = kwargs.get('character')

        self.save_interval = kwargs['save_interval']
        self.train_interval = train_interval
        self.update_interval = update_interval
        self.batch_size = batch_size
        self.train_after = train_after
        self.update_after = update_after
        self.use_per = use_per

        if buffer_kwargs is None:
            buffer_kwargs = {
                'size': buffer_size,
                'elements': {
                    'state': (model.observation_shape, model.injection_shape,
                              np.float32),
                    'action': (model.action_shape, np.float32),
                    'reward': ((1, ), np.float32),
                    'done': ((1, ), np.uint8),
                    'state_p': (model.observation_shape, model.injection_shape,
                                np.float32)
                }
            }

        super().__init__(model, train_interval, num_epochs, buffer_cls,
                         buffer_kwargs)

        self.buffer = CustomBuffer(**buffer_kwargs)

    def act(self, obs, action_space):
        # get the action from policy
        action = self.model.act(obs)
        action = action.detach().cpu().numpy()
        self.action_param = action

        return action

    def step(self, s, a, r, d, s_):
        # update critic1, critic2, actor, and the temperature
        self.curr_step += 1
        #if self.model.discrete:
        #    a = self.action_param
        self.collect(s, a, r, d, s_)
        info = {}
        if (self.curr_step % self.train_interval == 0
                and self.curr_step > self.train_after):
            info = self.train()
        if (self.curr_step % self.update_interval == 0
                and self.curr_step > self.update_after):
            self.model.update_trg()
        if self.curr_step % self.save_interval == 0 and self.model.is_save:
            save_dir = os.path.join(self.log_dir,
                                    f'ckpt/{int(self.curr_step/1000)}k')
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            self.model.save(save_dir)

        return info

    def train(self):
        if self.use_per:
            batch, weights = self.buffer.sample(self.batch_size)
        else:
            batch = self.buffer.sample(self.batch_size)
            weights = 1.
        import pdb
        pdb.set_trace()
        l_q1, l_q2, errors, mean_q1, mean_q2 = loss_func_cr(batch,
                                                            self.model,
                                                            weights=weights)
        l_pi, entropies = loss_func_ac(batch, self.model, weights=weights)
        entropy_loss = loss_func_alpha(entropies, self.model)

        self.model.q1.step(l_q1)
        self.model.q2.step(l_q2)
        self.model.pi.step(l_pi)

        self.model.alpha_optim.zero_grad()
        self.model.entropy_loss.backward()
        self.model.alpha_optim.step()

    def collect(self, obs, ac, rew, done, obs_):
        # Store given observations
        self.buffer.push(state=obs,
                         action=ac,
                         reward=rew,
                         done=done,
                         state_p=obs_)


class SACMAGENT_DISC(MAgent):
    def __init__(self):
        # TODO
        pass

    def train(self, **kwargs):
        # TODO
        pass

    def act(self, **kwargs):
        # TODO
        pass

    def train(self):
        # TODO
        pass


# Try out the agent
if __name__ == '__main__':
    m = SACModel
    m = SACAGent_DISC

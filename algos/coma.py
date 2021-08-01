from copy import deepcopy
from easydict import EasyDict
from utils import CustomEnvWrapper
from pommerman.configs import one_vs_one_env, team_competition_env
from functools import reduce
from typing import Callable, Iterable, List, Union
import os
import operator
import torch
import torch.nn.functional as F
import numpy as np
import gym
import pommerman
import marlenv

from rl2.models.torch.base import BranchModel, InjectiveBranchModel, TorchModel
from rl2.agents.base import MAgent, Agent
from rl2.buffers import ReplayBuffer


def loss_func_ac(data, model, **kwargs):
    s, a, r, d, s_, adv = tuple(
        map(lambda x: torch.from_numpy(x).float().to(model.device), data)
    )
    ac_dist, val_dist = model(s, hidden=kwargs.hidden, mask=d)
    nlps = -ac_dist.log_prob()
    loss = -nlps * adv.mean()

    return loss


def loss_func_cr(data, model, **kwargs):
    """
    loss function for centralized critic
    """
    s, a, r, d, s_, adv = tuple(
        map(lambda x: torch.from_numpy(x).float().to(model.device), data)
    )
    q_trg = None
    q = None
    loss = F.smooth_l1_loss(q_trg, q)

    return loss


def loss_func(data, model, **kwargs):
    actor_loss = loss_func_ac(data, model)
    critic_loss = loss_func_cr(data, model)

    return actor_loss, critic_loss


class COMAPGModel(TorchModel):
    def __init__(self,
                 observation_shape: Union[Iterable[int], Iterable[Iterable]],
                 action_shape,
                 joint_action_shape,
                 n_agents: int = 2,
                 encoder: torch.nn.Module = None,
                 encoded_dim: int = 64,
                 optimizer='torch.optim.Adam',
                 lr=1e-4,
                 discrete: bool = True,
                 deterministic: bool = False,
                 flatten: bool = False,  # True if you don't want CNN enc
                 reorder: bool = False,  # flag for (C, H, W)
                 recurrent: bool = True,
                 **kwargs):
        # Presetting for handling additional observation.
        self.additional = False
        if hasattr(observation_shape[0], '__iter__'):
            # Unpack the tuple of shapes
            observation_shape, add_shape = observation_shape
            # First shape of the observation assumed to be a locational info
            # and Thus will be treated as a main obs and Assummed to be (C,H,W)
            self.additional = True

        super().__init__(observation_shape, action_shape, **kwargs)
        if hasattr(encoder, 'output_shape'):
            encoded_dim = encoder.output_shape

        self.encoded_dim = encoded_dim
        self.recurrent = recurrent
        self.n_agents = n_agents
        # TODO: Currently recurrent unit uses LSTM -> change to GRU later
        # Actor uses inputs of locational obs, additional obs, and old action.
        # Policy network uses locational obs as main obs, and other inputs are
        # injected together after passing the CNN enc.
        self.policy = InjectiveBranchModel(observation_shape, action_shape,
                                           injection_shape=(9,),
                                           encoded_dim=encoded_dim,
                                           optimizer=optimizer,
                                           lr=lr,
                                           discrete=discrete,
                                           deterministic=deterministic,
                                           flatten=flatten,
                                           reorder=False,
                                           recurrent=True,  # True
                                           **kwargs)
        # Decentralized Actors
        self.policies = [deepcopy(self.policy) for i in range(self.n_agents)]

        # Centralized Critic
        self.value = InjectiveBranchModel(observation_shape,
                                          (action_shape[0],),
                                          (len(action_shape),),  # n_acs?
                                          encoder=encoder,
                                          encoded_dim=encoded_dim,
                                          optimizer=optimizer,
                                          lr=lr,
                                          discrete=True,
                                          deterministic=True,
                                          flatten=flatten,
                                          reorder=reorder,
                                          recurrent=False,
                                          **kwargs)

        # Initialize params
        list(map(self.init_params, self.policies))
        self.init_params(self.value)

    def forward(self, obs: torch.tensor, *args, **kwargs):
        obs = obs.to(self.device)
        args = [a.to(self.device) for a in args]
        action_dist = self.policy(obs, **kwargs)
        value_dist = self.value(obs, **kwargs)
        if self.recurrent:
            action_dist = action_dist[0]
            value_dist = value_dist[0]

        return action_dist, value_dist

    def act(self, obs: np.ndarray, agent: int, *args) -> np.ndarray:
        # *args contains the injections i.e. (additional obs, old_action)
        inj = reduce(operator.add, args)
        # FIXME: multiple injections might be handled in _infer_from_numpy
        policy = self.policies[agent]
        action_dist, hidden = self._infer_from_numpy(policy, obs, inj)
        action = action_dist.sample().squeeze()
        action = action.detach().cpu().numpy()

        log_prob = action_dist.log_prob(action)
        log_prob = log_prob.detach().cpu().numpy()

        info = {}
        info['lob_prob'] = log_prob
        if self.recurrent:
            info['hidden'] = hidden

        return action, info

    def val(self, state: np.ndarray, acs: np.ndarray, **kwargs) -> np.ndarray:
        agent_idx = kwargs.get('agent_idx')
        acs_rest = kwargs.get('acs_rest')

        val_dist = self.enc_global(obs)

        val_dist, hidden = self._infer_from_numpy(self.value, obs)
        value = val_dist.mean.squeeze()
        value = value.detach().cpu().numpy()

        info = {}
        if self.recurrent:
            info['hidden'] = hidden

        return value, info

    def save(self, save_dir):
        torch.save(self.state_dict(),
                   os.path.join(save_dir, type(self).__name__ + '.pt'))
        print(f'model saved in {save_dir}')

    def load(self, load_dir):
        ckpt = torch.load(load_dir, map_location=self.device)
        self.load_state_dict(ckpt)


class COMAPGAgent(Agent):
    def __init__(self,
                 model=TorchModel,
                 n_env=1,
                 buffer_cls: ReplayBuffer = ReplayBuffer,  # EpisodicBuffer
                 buffer_kwargs: dict = None,
                 batch_size: int = 128,
                 train_interval: int = 1,
                 num_epochs: int = 1,
                 loss_func: Callable = loss_func,
                 gamma: float = 0.99,
                 lamda: float = 0.95,
                 **kwargs):

        if 'config' in kwargs.keys():
            self.config = EasyDict(kwargs.get('config'))
        self.init_collect = self.config.init_collect
        self.train_interval = self.config.train_interval
        self.batch_size = self.config.batch_size
        self.gamma = self.config.gamma
        if buffer_kwargs is None:
            buffer_kwargs = {'size': self.config.buffer_size}

        super().__init__(model, train_interval, num_epochs,
                         buffer_cls, buffer_kwargs,)
        self.obs = None
        self.n_env = n_env
        if n_env == 1:
            self.done = False
        else:
            self.done = [False] * n_env
        self.gamma = gamma
        self.lamda = lamda
        self.batch_size = batch_size

        self.value = None
        self.nlp = None
        self.loss_func = loss_func

        self.model._init_hidden(self.done)
        self.hidden = self.model.hidden
        self.pre_hidden = self.hidden

    def act(self, obs, ac):
        action, info = self.model.act(obs, ac)
        self.nlp = -info['log_prob']
        self.value = self.model.val(obs)

        if self.model.recurrent:
            self.hidden = info['hidden']

        return action

    def step(self, s, a, r, d, s_):
        self.curr_step += 1
        self.collect(s, a, r, self.done, self.value, self.nlp)
        self.done = d

        if self.model.recurrent:
            self.model._update_hidden(d, self.hidden)

        info = {}
        if self.curr_step % self.train_interval == 0:
            value = self.model.val(s_)
            advs = self._calculate_adv(s, a, d, self.gamma, self.lamda)
            info = self.train(advs)
            self.buffer.reset()
            if self.model.recurrent:
                self.pre_hidden = self.model.hidden

        return info

    def _calculate_adv(self, state, joint_action, action):
        q_vals = self.model.critic_trg(state, joint_action)
        q_val = q_vals[action]
        action_dist = self.model.actor(state)
        counterfactual = torch.dot(action_dist, q_vals)
        adv = q_val - counterfactual

        return adv

    def train(self, advs, **kwargs):
        # data = self.buffer.sample(self.batch_size,
        #                           recurrent=self.model.recurrent)
        data = self.buffer.get(-1)  # get the last transition
        loss_ac, loss_cr = self.loss_func(data)
        self.model.policy.step(loss_ac)
        self.model.critic.step(loss_cr)

        info = {}
        info['actor_loss'] = loss_ac
        info['critic_loss'] = loss_cr

        return info

    def collect(self, *args):
        self.buffer.push(*args)


class COMAgent(MAgent):
    def __init__(self,
                 models: List[TorchModel],
                 train_interval: int = 1,
                 num_epochs: int = 1,
                 buffer_cls: ReplayBuffer = ReplayBuffer,
                 buffer_kwargs: dict = None,
                 n_agents: int = 2,
                 loss_func: Callable = loss_func,
                 **kwargs,
                 ):

        if 'config' in kwargs.keys():
            self.config = EasyDict(kwargs.get('config'))

        self.init_collect = self.config.init_collect
        self.train_interval = self.config.train_interval
        self.batch_size = self.config.batch_size
        self.gamma = self.config.gamma

        if buffer_kwargs is None:
            buffer_kwargs = {'size': self.config.buffer_size}

        super().__init__(models, train_interval, num_epochs,
                         buffer_cls, buffer_kwargs)
        self.model = model  # COMA uses a single model instance
        self.obs = None
        self.done = False

        self.n_agents = n_agents
        self.values = [None] * n_agents
        self.nlps = [None] * n_agents
        self.loss_func = loss_func

        self.model._init_hidden(self.done)
        self.hidden = self.model.hidden
        self.pre_hidden = self.hidden

    def act(self, joint_obs, old_joint_action) -> np.ndarray:
        joint_action = np.empty(self.n_agents)

        for i, (obs, old_action) in enumerate(zip(joint_obs, old_joint_action)):
            # FIXME: Handle cases for using a single observation
            loc_obs = obs.get('locational')
            add_obs = obs.get('additional')
            joint_action[i] = self.model.act(loc_obs, i, add_obs, old_action)

        return joint_action

    def collect(self, *args) -> 'Maybe Some statistics?':
        self.buffer.push(*args)

    def train(self) -> 'Maybe Train Result?':
        return super().train()


if __name__ == '__main__':
    config = team_competition_env()
    env = CustomEnvWrapper(config)

    # get partial obs
    obs = env.reset()
    print(obs)
    # get global obs
    out = env.get_global_obs()
    g_loc = np.array([d.get('locational') for d in out])
    g_add = np.array([d.get('additional') for d in out])

    # (5, 11, 11) = (len(loc) * vgrid * hgrid)
    # (5, 9, 9) = (len(loc) * (2*view_range+1) * (2*view_range+1))

    observation_shape = env.observation_shape
    print(observation_shape)
    ac_shape = (env.action_space.n,)
    n_agents = len(env._agents)
    jac_shape = tuple([ac_shape for i in range(n_agents)])

    model = COMAPGModel(observation_shape, ac_shape, jac_shape)
    agent_config = {
        'num_workers': 1,
        'buffer_size': 1,
        'batch_size': 1,
        'num_epochs': 1,
        'update_interval': 10,
        'train_interval': 10,
        'init_collect': 1,
        'log_interval': 1,
        'lr_ac': 1e-4,
        'lr_cr': 1e-3,
        'gamma': 0.95,
        'eps': 0.01,
        'polyak': 0.99
    }
    # agent = COMAPGAgent(model, config=agent_config)
    magent = COMAgent([model], config=agent_config)
    jac_init = np.zeros(magent.n_agents)
    magent.act(obs, jac_init)

    # ReplayBuffer(size=1, state_shape)

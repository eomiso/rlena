import gym
from gym.core import RewardWrapper
import numpy as np
from collections import deque

import torch as T

from typing import List
from rl2.agents.base import Agent
from rl2.workers.base import RolloutWorker, EpisodicWorker
from rl2.examples.temp_logger import Logger


class SimpleWorker:
    def __init__(
            self,
            env: gym.Env,
            n_env: int = 1,
            agents: [Agent] = [],  # training agents
            max_step=500,
            max_episode: int = 1e8,
            training: bool = True,
            render: bool = False,
            random_until: int = 200,
            **kwargs):
        self.env = env
        self.n_env = n_env
        self.trainee_agents = agents
        self.random_until = random_until
        if not training:
            self.random_until = 0
        self.training = training

        self.max_episode = max_episode
        self.episode_cnt = 0
        self.max_step = max_step
        self.render = render

    def run(self):
        with T.autograd.set_detect_anomaly(True):
            while self.episode_cnt < self.max_episode:
                self.rollout()
                self.episode_cnt += 1

    def rollout(self):
        step_cnt = 0
        obs = self.env.reset()
        done = False
        if self.render:
            self.env.render()

        while not done and step_cnt < self.max_step:
            action_dist = [None] * len(self.env._agents)
            actions = self.env.act(obs)
            for agent in self.trainee_agents:
                agent_id = agent.agent_id
                actions[agent_id] = agent.act(obs[agent_id])
                #print('actions: {}'.format(actions))

                # the buffer should save the action discributions
                dist, _, _ = agent.model(obs[agent_id])
                action_dist[agent_id] = dist.probs.detach()[0].cpu().numpy()

            if self.render:
                self.env.render()
            obs_, rew, done, info = self.env.step(actions)

            if self.training:
                for agent in self.trainee_agents:
                    agent_id = agent.agent_id
                    agent.step(obs[agent_id], action_dist[agent_id],
                               rew[agent_id], done, obs_[agent_id])

            obs = obs_
            step_cnt += 1
            #if done:
            #    print('Episode {} finished'.format(self.episode_cnt))
        return done


class SACDworker(EpisodicWorker):
    def __init__(self, env, n_env, agents, n_agents, max_episodes: int,
                 log_interval: int, logger, render, render_interval, random_until, **kwargs):
        super().__init__(env,
                         n_env,
                         agents,
                         max_episodes=max_episodes,
                         log_interval=log_interval,
                         logger=logger,
                         **kwargs)
        self.render = render
        self.render_interval = render_interval
        self.done = False
        self.winner = deque(maxlen=10)
        self.trainee_agents = agents
        self.env = env
        self.obs = self.env.reset()
        self.random_until = random_until
        self.agents = agents
        self.n_agents = n_agents
        self.render_mode = 'rgb_array'

    def rollout(self):
        action_dist = [None] * len(self.env._agents)
        actions = self.env.act(self.obs)  # (None:trainee_agents, action: simple_agent)

        for agent in self.trainee_agents:
            i = agent.agent_id
            actions[i] = agent.act(self.obs[i])
            #print('actions: {}'.format(actions))

            # the buffer should save the action discributions
            dist, _, _ = agent.model(self.obs[i])
            action_dist[i] = dist.probs.detach()[0].cpu().numpy()

        obs_, rewards, done, info = self.env.step(actions)
        if rewards[0] == 0:
            rewards[0] = 0.3 
        if rewards[0] == -1:
            rewards[0] = -5 
        self.done = done if np.asarray(done).size == 1 else any(done)

        if self.training:
            for agent in self.trainee_agents:
                i = agent.agent_id
                info_a = agent.step(self.obs[i], action_dist[i], rewards[i],
                                    done, obs_[i])
                for k, v in info_a.items():
                    if isinstance(v, tuple):
                        info[k] = sum(info_a[k])
                    else:
                        info[k] = info_a[k]
            self.num_steps += self.n_env
            #self.logger.scalar_summary(info_a, self.num_steps)
            self.episode_score = self.episode_score + np.array(rewards)
            steps = ~np.asarray(done)
            self.ep_steps += steps.astype(np.int)

        if self.done:
            self.num_episodes += 1
            obs = self.env.reset()
            self.scores.append(self.episode_score)
            self.episode_score = np.zeros_like(self.episode_score, np.float)
            self.ep_length.append(self.ep_steps)
            self.ep_steps = np.zeros_like(self.ep_steps)
            self.winner.append(info.pop('result').value)
        self.obs = obs_
        results = None
        return self.done, info, results

    def run(self):
        while self.num_episodes < self.max_episodes:
            prev_num_ep = self.num_episodes
            done, info_r, results = self.rollout()

            if self.render and self.start_log_image:
                image = self.env.render(self.render_mode)
                self.logger.store_rgb(image)

            log_cond = done if np.asarray(done).size == 1 else any(done)
            if log_cond:
                if self.start_log_image:
                    print('save_video')
                    self.logger.video_summary(tag='playback',
                                              step=self.num_steps)
                    self.start_log_image = False
                if self.render:
                    if (prev_num_ep // self.render_interval !=
                            self.num_episodes // self.render_interval):
                        self.start_log_image = True
                ep_lengths = np.asarray(self.ep_length)

                info = {
                    'Counts/num_steps': self.num_steps,
                    'Counts/num_episodes': self.num_episodes,
                    'Episodic/ep_length': ep_lengths.max(-1).mean(),
                    'Episodic/avg_winrate': np.asarray(self.winner).mean()
                }
                if self.n_agents == 4:
                    ep_lengths = np.split(ep_lengths, 2, axis=-1)
                scores = np.split(np.asarray(self.scores), 2, axis=-1)

                for team, score in enumerate(scores):
                    info[f'team{team}/rews_avg'] = np.mean(score)
                    if self.n_agents == 4:
                        for i, ep_len in enumerate(ep_lengths[team].mean(0)):
                            info[f'team{team}/agent{i}_ep_length'] = ep_len
                    else:
                        info[
                            f'team{team}/agent{team}_ep_length'] = ep_lengths.mean(
                            )

                self.info.update(info)

                if info_r.get('entropies', False):
                    rm_keys = ['winners', 'original_obs', 'done', 'is_loss']
                    for key in rm_keys:
                        info_r.pop(key, None)
                    self.info.update(info_r)
                if (prev_num_ep // self.log_interval) != (self.num_episodes //
                                                          self.log_interval):
                    self.logger.scalar_summary(self.info, self.num_steps)


from algos.sac_discrete import SACAgentDISC, SACModelDISC
from utils.wrapper import ConservativeEnvWrapper
from easydict import EasyDict
from pommerman.characters import Bomber
from pommerman.agents.simple_agent import SimpleAgent
from pommerman.configs import one_vs_one_env
from rl2.buffers.base import ReplayBuffer

if __name__ == "__main__":
    myconfig = {
        'gamma': 0.99,
        'train_interval': 5,
        'train_after': 1024,
        'update_interval': 30,
        'update_after': 1024,
        'save_interval': 1000,
        'batch_size': 128,
        'random_until': 1024,
        'render': True,
    }

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
    config = EasyDict(myconfig)

    env_config = one_vs_one_env()
    env_config['env_kwargs']['agent_view_size'] = 4
    env_config['env_kwargs']['max_step'] = 1000
    env = ConservativeEnvWrapper(env_config)
    action_shape = env.action_space.n if hasattr(
        env.action_space, 'n') else env.action_space.shape

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

    trainee_agent = SACAgentDISC(model,
                                 batch_size=config.batch_size,
                                 train_after=config.train_after,
                                 train_interval=config.train_interval,
                                 update_interval=config.update_interval,
                                 save_interval=100,
                                 buffer_cls=ReplayBuffer,
                                 buffer_kwargs=buffer_kwargs,
                                 save_dir='sac_discrete/test/ckpt',
                                 log_dir='sac_discrete/test/log',
                                 character=Bomber(0, env_config["game_type"]))

    agents = {
        0: trainee_agent,
        1: SimpleAgent(env_config['agent'](1, env_config["game_type"])),
    }

    env.set_init_game_state(None)
    env.set_agents(list(agents.values()))
    env.set_training_agents(0)
    env.seed(44)
    config = EasyDict({
        'n_agents': 2,
        'n_env': 1,
        'batch_size': 128,
        'num_epochs': 1,
        'update_interval': 128,
        'train_interval': 30,
        'init_collect': 1,
        'log_interval': 10,
        'lr_ac': 1e-4,
        'lr_cr': 1e-3,
        'gamma': 0.95,
        'lamda': 0.99,
        'eps': 0.01,
        'polyak': 0.99
    })
    logger = Logger(name='TestSACD', args=config)
    worker = SACDworker(env,
                        agents=[trainee_agent],
                        n_agents=config.n_agents,
                        n_env=config.n_env,
                        max_episodes=20000,
                        training=True,
                        logger=logger,
                        log_interval=config.log_interval,
                        render=True,
                        max_step=1000,
                        random_until=400,
                        is_save=True)

    import pdb
    pdb.set_trace()
    worker.run()

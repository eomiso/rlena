import gym

import torch as T

from typing import List
from rl2.agents.base import Agent
from rl2.workers.base import RolloutWorker


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
                actions[agent_id] = agent.act(obs[agent_id],
                                              rand_until=self.random_until)
                print('actions: {}'.format(actions))

                # the buffer should save the action discributions
                dist, _, _ = agent.model(obs[agent_id])
                action_dist[agent_id] = dist.probs.detach()[0].numpy()

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
            if done:
                print('Episode {} finished'.format(self.episode_cnt))

import os
import imageio
import numpy as np

from pathlib import Path
from typing import Dict, List
from pommerman.agents import BaseAgent
from pommerman.envs.v0 import Pomme
from pommerman.configs import team_competition_env


class CustomAgent(BaseAgent):
    def act(self, *args):
        pass


class CustomEnvWrapper(Pomme):
    def __init__(self, config) -> None:
        super().__init__(**config['env_kwargs'])
        self.seed(0)
        agents = {}
        for agent_id in range(4):
            agents[agent_id] = CustomAgent(
                config["agent"](agent_id, config["game_type"]))
        self.set_agents(list(agents.values()))
        self.set_init_game_state(None)
        view_range = 2 * self._agent_view_size + 1
        locational_shape = (5, view_range, view_range)
        additional_shape = (8,)
        self.observation_shape = (locational_shape, additional_shape)

    def reset(self):
        obs = super().reset()
        obs = self._preprocessing(obs)

        return obs

    def step(self, acs):
        obs, reward, done, info = super().step(acs)
        info['original_obs'] = obs
        obs = self._preprocessing(obs)

        return obs, reward, done, info

    def get_global_obs(self):
        obs = self.model.get_observations(curr_board=self._board,
                                          agents=self._agents,
                                          bombs=self._bombs,
                                          flames=self._flames,
                                          is_partially_observable=False,
                                          agent_view_size=self._agent_view_size,
                                          game_type=self._game_type,
                                          game_env=self._env)
        obs = self._preprocessing(obs)

        return obs

    def _preprocessing(self, obs: List[Dict], **kwargs) -> List[Dict]:
        out = []
        for d in obs:
            custom_obs = {}
            keys = ['alive', 'game_type', 'game_env']
            _ = list(map(d.pop, keys))  # remove useless obs

            # Change enums into int
            d.update({'teammate': d.get('teammate').value})
            enemies = list(map(lambda x: x.value, d.get('enemies')))
            enemies.remove(9)  # Remove dummy agent from enemies list
            d.update({'enemies': enemies})

            # Gather infos
            locational = []
            additional = []
            for k, v in d.items():
                if hasattr(v, 'shape'):
                    # Make border walls for locational obs
                    # obs['board'] borders are represented as 1(= Rigid wall)
                    # else borders are filled with 0 values.
                    if k != 'board':
                        v = np.insert(v, (0, v.shape[0]), 0, axis=0)
                        v = np.insert(v, (0, v.shape[1]), 0, axis=1)

                        if not kwargs.setdefault('global_obs', False):
                            for _ in range(self._agent_view_size - 1):
                                v = np.insert(v, (0, v.shape[0]), 0, axis=0)
                                v = np.insert(v, (0, v.shape[1]), 0, axis=1)

                    else:
                        v = np.insert(v, (0, v.shape[0]), 1, axis=0)
                        v = np.insert(v, (0, v.shape[1]), 1, axis=1)

                        if not kwargs.get('global_obs'):
                            for _ in range(self._agent_view_size - 1):
                                v = np.insert(v, (0, v.shape[0]), 1, axis=0)
                                v = np.insert(v, (0, v.shape[1]), 1, axis=1)

                    # Cut views by centering the agent for localized observation
                    if not kwargs.get('global_obs'):
                        pos = np.array(d.get('position'))
                        view_range = 2 * self._agent_view_size + 1
                        v = v[pos[0]:pos[0] + view_range,
                              pos[1]:pos[1] + view_range]
                    locational.append(v)

                    if k == 'board' and kwargs.get('onehot', False):
                        locational.pop()
                        # one hot vectorized board observation
                        for i in range(14):
                            onehot = np.asarray(v == i, dtype=np.int)
                            locational.append(onehot)

                else:
                    if hasattr(v, '__iter__'):
                        additional += v
                    else:
                        additional.append(v)

            custom_obs.update({
                'locational': np.stack(locational),
                'additional': np.array(additional, dtype='float64')
            })

            out.append(custom_obs)

        return out


class TwoVsTwoPomme(CustomEnvWrapper):
    def __init__(self, **kwargs):
        config = team_competition_env()
        config['env_kwargs'].update({
            'max_steps': kwargs.get('max_steps'),
        })
        super().__init__(config)
        # Reorder outputs from env.reset & env.step
        self.order = np.array([0, 2, 1, 3])
        self.random_num_wall = kwargs.get('random_num_wall', True)
        if self.random_num_wall:
            self.max_rigid = kwargs.get('max_rigid', 18)
            self.max_wood = kwargs.get('max_wood', 8)
        self.remove_stop = int(kwargs.get('remove_stop', False))
        self.onehot = kwargs.get('onehot', False)
        if self.onehot:
            self.observation_shape = ((18, 9, 9), (8,))

    def step(self, acs):
        acs = acs.copy() + self.remove_stop
        obs, reward, done, info = Pomme.step(self, acs)
        info.update({'original_obs': obs,
                     'done': done})
        obs = self._preprocessing(obs, onehot=self.onehot)
        dones = list(map(lambda x: not x.is_alive, self._agents))

        obs, reward, dones = self._reorder(obs, reward, dones)
        dead_agents = np.where(self.old_dones != dones)[0]
        if len(dead_agents) > 0:
            reward = np.asarray(reward, dtype=np.float)
            reward[dead_agents] -= 0.5
            reward = list(reward)
        self.old_dones = np.asarray(dones)
        dones.append(done)
        obs.append(self.get_global_obs())

        if done:
            if self._step_count > self._max_steps:
                reward = list(np.asarray(reward, dtype=np.float) - 1.0)
            dones = [True] * 5
        return obs, reward, dones, info

    def _reorder(self, *args):
        out = []
        for a in args:
            a = list(np.array(a)[self.order])
            out.append(a)

        return out

    def get_global_obs_shape(self):
        g_obs = self.get_global_obs().values()
        self.g_obs_shape = [o.shape for o in g_obs]

        return self.g_obs_shape

    def get_global_obs(self):
        obs = self.model.get_observations(curr_board=self._board,
                                          agents=self._agents,
                                          bombs=self._bombs,
                                          flames=self._flames,
                                          is_partially_observable=False,
                                          agent_view_size=self._agent_view_size,
                                          game_type=self._game_type,
                                          game_env=self._env)

        obs = self._preprocessing(obs, global_obs=True, onehot=self.onehot)
        # locational obs only uses the first obs
        loc = obs[0].get('locational')
        # additional obs concatenate all values
        adds = list(map(lambda x: x.get('additional'), obs))
        add = np.concatenate(adds)

        out = {'locational': loc,
               'additional': add}

        return out

    def reset(self):
        # original = 36,36
        # wood min = 20; probably due to the num_items
        if self.random_num_wall:
            self._num_rigid = np.random.randint(0, self.max_rigid) * 2
            self._num_wood = 20 + (np.random.randint(0, self.max_wood) * 2)
        obs = Pomme.reset(self)
        obs = self._preprocessing(obs, onehot=self.onehot)
        obs.append(self.get_global_obs())

        self.old_dones = np.asarray([False] * 4)

        return obs

    def render(self, *args, **kwargs):
        if args[0] == 'human':
            path = './tmp/'
            Path(path).mkdir(parents=True, exist_ok=True)
            Pomme.render(self, record_pngs_dir=path, *args, **kwargs)
            filename = os.listdir(path)[-1]
            rgb_array = imageio.imread(
                os.path.join(path, filename), pilmode='RGB')
            os.remove(os.path.join(path, filename))
            Path(path).rmdir()

            return rgb_array
        else:
            return Pomme.render(self, *args, **kwargs)

from typing import Dict, List
import numpy as np
from pommerman.agents import BaseAgent
from pommerman.envs.v0 import Pomme
from algos.utils import CustomEnvWrapper


class ConservativeEnvWrapper(CustomEnvWrapper):
    """
    A very similar wrapper to Eunki's CustomEnvWrapper. Made change in or added
      - Option for OneVsOne env
      - functionality for multiple training agents
      - maintain original observations for non-custom agents(SimpleAgent, RandomAgent)
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        if 'OneVsOne-v0' in config['env_id']:
            self.one_vs_one = True
        else:
            self.one_vs_one = False

        self._training_agents = []

    def set_training_agents(self, *args):
        self._training_agents = list(args)

    def act(self, obs):
        agents = [
            agent for agent in self._agents
            if agent.agent_id not in self._training_agents
        ]

        # the env model uses the agent's sepcific observation
        # indexed by agent_id
        actions = self.model.act(agents, obs, self.action_space)
        for i, action in enumerate(actions):
            if i in self._training_agents:
                actions.insert(i, None)
        return actions

    def _preprocessing(self, obs: List[Dict], **kwargs) -> List[Dict]:
        out = []
        for i, d in enumerate(obs):
            if i not in self._training_agents:
                out.append(d)
                continue
            custom_obs = {}
            keys = ['alive', 'game_type', 'game_env']
            _ = list(map(d.pop, keys))  # remove useless obs

            # Change enums into int
            d.update({'teammate': d.get('teammate').value})
            enemies = list(map(lambda x: x.value, d.get('enemies')))
            if not self.one_vs_one:
                enemies.remove(9)  # Remove dummy agent from enemies list
            d.update({'enemies': enemies})

            # Gather infos
            locational = []
            additional = []
            for k, v in d.items():
                if hasattr(v, 'shape'):
                    # Make border walls for locational obs
                    # obs['board'] borders are represented as 2(= Rigid wall)
                    # else borders are filled with 0 values.
                    if k != 'board':
                        for _ in range(self._agent_view_size):
                            v = np.insert(v, (0, v.shape[0]), 0, axis=0)
                            v = np.insert(v, (0, v.shape[1]), 0, axis=1)
                    else:
                        for _ in range(self._agent_view_size):
                            v = np.insert(v, (0, v.shape[0]), 1, axis=0)
                            v = np.insert(v, (0, v.shape[1]), 1, axis=1)

                    # Cut views by centering the agent for localized observation
                    if not kwargs.setdefault('global', False):
                        pos = np.array(d.get('position'))
                        view_range = 2 * self._agent_view_size + 1
                        v = v[pos[0]:pos[0] + view_range,
                              pos[1]:pos[1] + view_range]

                    locational.append(v)

                else:
                    if hasattr(v, '__iter__'):
                        additional += v
                    else:
                        additional.append(v)

            custom_obs.update({
                'locational':
                np.stack(locational),
                'additional':
                np.array(additional, dtype='float64')
            })

            out.append(custom_obs)

        return out

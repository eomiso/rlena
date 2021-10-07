import numpy as np
import pprint

from collections import deque
from rl2.buffers.base import ReplayBuffer


class EpisodicBuffer(ReplayBuffer):
    def __init__(self, size=1, n_env=1, elements=None, max_episodes=30):
        if elements is None:
            elements = [
                'loc',
                'add',
                'hidden',
                'actions',
                'rewards',
                'dones',
                'values',
                'g_loc',
                'g_add',
            ]
        self.episodes = deque(maxlen=max_episodes)
        self.curr_episode = 0
        self.max_episodes = max_episodes
        super().__init__(size, elements)

    def __call__(self, idx=None):
        if idx is None:
            return list(self.episodes) + self.to_dict()
        else:
            return self.episodes[idx]

    def __repr__(self) -> str:
        if self.curr_episode == 1:
            out = ''
            for key, value in self.to_dict().items():
                if isinstance(value, np.ndarray):
                    value = value.shape
                out += key + '\n' + pprint.pformat(value) + '\n'
            return out
        else:
            return pprint.pformat(self.episodes)

    def reset(self):
        self.curr_episode += 1
        if self.curr_episode > 1:
            self.episodes.append(self.to_dict())
        super().reset()

    def push(self, *args):
        kwargs = dict(zip(self.keys, args))
        super().push(**kwargs)

    def sample(self, *args):
        out = []
        if len(args) == 0:
            args = self.keys
        for key in args:
            values = [d[key] for d in self.episodes]
            out.append(np.vstack(values))

        return out

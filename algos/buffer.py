from rl2.buffers.base import ReplayBuffer
from rl2.buffers.prioritized import PrioritizedReplayBuffer
from collections import Iterable
import numpy as np


class CustomBuffer(ReplayBuffer):
    def __int__(self, capacity=None, **kwargs):
        super().__init__(**kwargs)
        self.reset()

    def reset(self):
        if self.specs is not None:
            for k, shape_dtype in zip(self.keys, self.specs):
                if len(shape_dtype) == 2:
                    shape, dtype = shape_dtype
                elif len(shape_dtype) == 3:
                    shape1, shape2, dtype = shape_dtype
                else:
                    shape = shape_dtype
                    dtype = None
                try:
                    assert isinstance(shape,
                                      Iterable), 'Non-iterable shape given'
                except UnboundLocalError:
                    assert isinstance(shape1, Iterable) and isinstance(
                        shape2, Iterable), 'Non-iterable shape given'
                if dtype and dtype.__module__ == 'numpy':
                    setattr(
                        self, k, self.max_size * [{
                            'locational':
                            np.ones(tuple(shape1), dtype=dtype),
                            'additional_shape':
                            np.ones(tuple(shape2), dtype=dtype)
                        }])
                else:
                    setattr(self, k, [None] * self.max_size)
        else:
            for k in self.keys:
                setattr(self, k, [None] * self.max_size)

        self.curr_idx = 0
        self.curr_size = 0

    def push(self, **kwargs):
        for key in self.keys:
            assert key in kwargs.keys()
            if key == 'state':
                getattr(self, key)[
                    self.curr_idx]['locational'] = kwargs[key]['locational']
                getattr(self, key)[
                    self.curr_idx]['additional'] = kwargs[key]['additional']
            else:
                getattr(self, key)[self.curr_idx] = kwargs[key]
        self.curr_size = min(self.curr_size + 1, self.max_size)
        self.curr_idx = (self.curr_idx + 1) % self.max_size


from easydict import EasyDict

if __name__ == "__main__":
    model = {
        'observation_shape': (1, 2, 3),
        'injection_shape': (8, ),
        'action_shape': (6, )
    }
    model = EasyDict(model)
    buffer_size = 100
    buffer_kwargs = {
        'capacity': buffer_size,
        'elements': {
            'state':
            (model.observation_shape, model.injection_shape, np.float32),
            'action': (model.action_shape, np.float32),
            'reward': ((1, ), np.float32),
            'done': ((1, ), np.uint8),
            'state_p':
            (model.observation_shape, model.injection_shape, np.float32)
        }
    }
    buff = CustomBuffer(**buffer_kwargs)
    buff.reset()

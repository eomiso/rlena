import torch
import torch.nn.functional as F
from rl2.models.torch.base import InjectiveBranchModel
import numpy as np


class IBMWithNormalization(InjectiveBranchModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = np.finfo(np.float32).eps.item()

    def foward(self, observation, injection, *args, **kwargs):
        observation = self._handle_obs_shape(observation)
        injection = injection.unsqueeze(0)
        injection = F.normalize(injection, eps=self.eps)
        ir = self.encoder(observation, *args, **kwargs)
        if self.recurrent:
            ir, hidden = ir
        ir = torch.cat([ir, injection], dim=-1)
        output = self.head(ir)

        if self.recurrent:
            return output, hidden
        return output

    def forward_trg(self, observation, injection, *args, **kwargs):
        observation = self._handle_obs_shape(observation)
        injection = F.normalize(injection, eps=self.eps)
        with torch.no_grad():
            ir = self.encoder_target(observation, *args)
            if self.recurrent:
                hidden = ir[1]
                ir = ir[0]
            ir = torch.cat([ir, injection], dim=-1)
            output = self.head_target(ir)

        if self.recurrent:
            return output, hidden
        return output

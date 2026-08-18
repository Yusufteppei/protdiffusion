import torch.nn as nn
from protdiffusion.config import d_res

class BackboneUpdate(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(d_res, 6)

    def forward(self, s):
        # s: [B, L, d_res]
        update = self.linear(s)

        rot_update = update[..., :3]
        trans_update = update[..., 3:]

        return rot_update, trans_update
import torch.nn as nn
from protdiffusion.config import d_res
from torch import Tensor
from jaxtyping import Float, Int, Bool, jaxtyped
from beartype import beartype


class BackboneUpdate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_res, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 6)
        )

    @jaxtyped(typechecker=beartype)
    def forward(self, s: Float[Tensor, "B L d_res"]) -> tuple[Float[Tensor, "B L 3"], Float[Tensor, "B L 3"]]:
        # s: [B, L, d_res]
        update = self.net(s)

        rot_update = update[..., :3]
        trans_update = update[..., 3:]

        return rot_update, trans_update
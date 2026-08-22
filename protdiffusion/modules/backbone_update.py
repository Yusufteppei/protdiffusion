import torch.nn as nn
from protdiffusion.config import d_res
from torch import Tensor
from jaxtyping import Float, Int, Bool, jaxtyped
from beartype import beartype


class BackboneUpdate(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(d_res, 6)

    @jaxtyped(typechecker=beartype)
    def forward(self, s: Float[Tensor, "B L d_res"]) -> tuple[Float[Tensor, "B L 3"], Float[Tensor, "B L 3"]]:
        # s: [B, L, d_res]
        update = self.linear(s)

        rot_update = update[..., :3]
        trans_update = update[..., 3:]

        return rot_update, trans_update
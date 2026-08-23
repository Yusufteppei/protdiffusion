import torch
import torch.nn as nn
import math
from protdiffusion.config import d_time, max_period, d_res
from jaxtyping import Float, Int, Bool, jaxtyped
from beartype import beartype


class TimeEmbedding(nn.Module):

    def __init__(self, d_time=d_time, max_period=max_period):
        super().__init__()

        assert d_time % 2 == 0, "d_time must be even"

        self.d_time = d_time

        half = d_time // 2

        frequencies = torch.exp(
            -math.log(max_period) *
            torch.arange(half) / half
        )

        self.register_buffer("frequencies", frequencies)
        self.translation_proj = nn.Linear(d_time, 3)
        self.rotation_proj = nn.Linear(d_time, 3)

    @jaxtyped(typechecker=beartype)
    def forward(self, timestep: Int[torch.Tensor, "B"]) -> tuple[Float[torch.Tensor, "B 3"], 
                                                                   Float[torch.Tensor, "B 3"]]:
        """
        timestep: [B]

        returns:
            [B, d_time]
        """

        timestep = timestep.float()

        args = timestep[:, None] * self.frequencies[None, :]

        #print(f"FREQ: {self.frequencies}")
        #print("ARGS", args)

        embedding = torch.cat(
            [
                torch.sin(args),
                torch.cos(args)
            ],
            dim=-1
        )

        rotation_time, translation_time = self.rotation_proj(embedding), self.translation_proj(embedding)

        return rotation_time, translation_time 
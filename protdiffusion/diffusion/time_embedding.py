import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):

    def __init__(self, d_time=128, max_period=10000):
        super().__init__()

        self.d_time = d_time

        half = d_time // 2

        frequencies = torch.exp(
            -math.log(max_period) *
            torch.arange(half) / half
        )

        self.register_buffer("frequencies", frequencies)

    def forward(self, timestep):
        """
        timestep: [B]

        returns:
            [B, d_time]
        """

        timestep = timestep.float()

        args = timestep[:, None] * self.frequencies[None, :]

        embedding = torch.cat(
            [
                torch.sin(args),
                torch.cos(args)
            ],
            dim=-1
        )

        return embedding
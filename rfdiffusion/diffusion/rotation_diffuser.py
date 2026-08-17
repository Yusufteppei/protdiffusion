import torch
import torch.nn as nn


class RotationDiffuser(nn.Module):

    def __init__(self, num_timesteps):
        super().__init__()

        self.num_timesteps = num_timesteps        


    def forward(self, x0, timestep, noise=None):

        return x0, noise
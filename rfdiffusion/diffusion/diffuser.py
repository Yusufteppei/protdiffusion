import torch
import torch.nn as nn
from rfdiffusion.diffusion import TranslationDiffusion, RotationDiffuser


class Diffuser(nn.Module):
    def __init__(self, timesteps=200):
        super().__init__()

        self.translation = TranslationDiffusion(timesteps=timesteps)
        self.rotation = RotationDiffuser(timesteps=timesteps)
        #self.T = torch.randint(timesteps, (B, ))

    def forward(self, t):

        trans = self.translation(x0, t, noise)
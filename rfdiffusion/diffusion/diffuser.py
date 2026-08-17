import torch
import torch.nn as nn
from rfdiffusion.diffusion import TranslationDiffuser, RotationDiffuser
from rfdiffusion.geometry import Rigid


class Diffuser(nn.Module):
    def __init__(self, num_timesteps=200):
        super().__init__()

        self.num_timesteps = num_timesteps
        self.translation = TranslationDiffuser(num_timesteps=num_timesteps)
        self.rotation = RotationDiffuser(num_timesteps=num_timesteps)

    def forward(self, x0: Rigid, timestep):
        x0_r, x0_t = x0.rotation, x0.translation
        
        xt_tr, noise_tr = self.translation(x0_t, timestep)
        xt_r, noise_r = self.rotation(x0_r, timestep)

        xt, noise_t = Rigid(rotation=xt_r, translation=xt_tr), Rigid(rotation=noise_r, translation=noise_tr)

        return xt, noise_t
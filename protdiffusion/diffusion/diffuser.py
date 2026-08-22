from torch import Tensor
import torch.nn as nn
from jaxtyping import Float, Bool, Int
from protdiffusion.diffusion import TranslationDiffuser, RotationDiffuser, TimeEmbedding
from protdiffusion.geometry import Rigid


class Diffuser(nn.Module):
    def __init__(self, num_timesteps=200):
        super().__init__()

        self.num_timesteps = num_timesteps
        self.translation_diffuser = TranslationDiffuser(num_timesteps=num_timesteps)
        self.rotation_diffuser = RotationDiffuser(num_timesteps=num_timesteps)

    def forward(self, x0: Rigid, timestep: Int[Tensor, "B"], 
                mask: Bool[Tensor, "B L"]) -> tuple[Rigid, Rigid]:
        
        x0_r, x0_t = x0.rotation, x0.translation
        
        xt_tr, noise_tr = self.translation_diffuser(x0_t, timestep, mask=mask)
        xt_r, noise_r = self.rotation_diffuser(x0_r, timestep, mask=mask)

        xt, noise_t = Rigid(rotation=xt_r, translation=xt_tr), Rigid(rotation_vector=noise_r, translation=noise_tr)

        return xt, noise_t
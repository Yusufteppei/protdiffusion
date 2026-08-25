from torch import Tensor
import torch.nn as nn
from jaxtyping import Float, Bool, Int, jaxtyped
from protdiffusion.diffusion import TranslationDiffuser, RotationDiffuser, TimeEmbedding
from protdiffusion.geometry import Rigid
from beartype import beartype


class Diffuser(nn.Module):
    def __init__(self, num_timesteps=200):
        super().__init__()

        self.num_timesteps = num_timesteps
        self.translation_diffuser = TranslationDiffuser(num_timesteps=num_timesteps)
        self.rotation_diffuser = RotationDiffuser(num_timesteps=num_timesteps)


    @jaxtyped(typechecker=beartype)
    def forward(self, rigid_0: Rigid, timestep: Int[Tensor, "B"], 
                mask: Bool[Tensor, "B L"]) -> tuple[Rigid, Rigid]:
        
        rigid_0_r, rigid_0_t = rigid_0.rotation, rigid_0.translation
        
        rigid_t_tr, noise_tr = self.translation_diffuser(rigid_0_t, timestep, mask=mask)
        rigid_t_r, noise_r = self.rotation_diffuser(rigid_0_r, timestep, mask=mask)

        xt, noise_t = Rigid(rotation=rigid_t_r, translation=rigid_t_tr), Rigid(rotation_vector=noise_r, translation=noise_tr)

        return xt, noise_t
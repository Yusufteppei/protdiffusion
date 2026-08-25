import torch
import torch.nn as nn
from jaxtyping import Float, jaxtyped, Bool, Int
from beartype import beartype


class TranslationDiffuser(nn.Module):

    def __init__(self, num_timesteps, beta_start=1e-3, beta_end=2e-2):

        super().__init__()

        beta = torch.linspace(beta_start, beta_end, num_timesteps)
        alpha = 1 - beta

        alpha_bar = torch.cumprod(alpha, dim=0)


        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    @jaxtyped(typechecker=beartype)
    def forward(self, trans_0: Float[torch.Tensor, "B L 3"], timestep: Int[torch.Tensor, "B"], 
                noise: Float[torch.Tensor, "B L 3"] = None,
                mask: Bool[torch.Tensor, "B L"] = None) -> tuple[Float[torch.Tensor, "B L 3"], 
                                                                  Float[torch.Tensor, "B L 3"]]:
        
        """
        trans_0: (..., 3)
        timestep:  (B,)
        """
        if noise is None:
            noise = torch.randn_like(trans_0)

        if mask is None:
            mask = torch.ones_like(trans_0[..., 0], dtype=torch.bool)

        alpha_bar = self.alpha_bar[timestep]

        while alpha_bar.ndim < trans_0.ndim:
            alpha_bar = alpha_bar.unsqueeze(-1)

        mask_3d = mask.unsqueeze(-1)

        trans_t = (
            alpha_bar.sqrt() * trans_0
            + (1.0 - alpha_bar).sqrt() * noise
        ) * mask_3d

        noise = noise * mask_3d
        assert torch.all(trans_t[~mask] == 0)
        assert torch.all(noise[~mask] == 0)
        return trans_t, noise
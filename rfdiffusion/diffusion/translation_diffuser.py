import torch
import torch.nn as nn



class TranslationDiffuser(nn.Module):

    def __init__(self, num_timesteps, beta_start=1e-4, beta_end=2e-2):

        super().__init__()

        beta = torch.linspace(beta_start, beta_end, num_timesteps)
        alpha = 1 - beta

        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)


    def forward(self, x0, timestep, noise=None):
        """
        x0: (..., 3)
        timestep:  (B,)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        alpha_bar = self.alpha_bar[timestep]

        while alpha_bar.ndim < x0.ndim:
            alpha_bar = alpha_bar.unsqueeze(-1)

        xt = (
            alpha_bar.sqrt() * x0
            + (1.0 - alpha_bar).sqrt() * noise
        )

        return xt, noise
import torch.nn as nn
from protdiffusion.modules import InputEmbedder, Trunk
from protdiffusion.diffusion import Diffuser, NoisePredictor
from protdiffusion.geometry import Rigid


class ProtDiffusion(nn.Module):
    def __init__(self, trunks, max_residues=2048):
        super().__init__()

        self.max_residues = max_residues
        self.input_embedder = InputEmbedder(self.max_residues)
        self.diffuser = Diffuser(num_timesteps=200)
        self.trunks = trunks
        self.trunk = Trunk()
        self.noise_predictor = NoisePredictor()

    def forward(self, tokens, mask, rigids, timestep) -> tuple[Rigid, Rigid]:
        single, pair = self.input_embedder(tokens)
        xt, noise_t = self.diffuser(x0=rigids, timestep=timestep)
        
        for _ in range(self.trunks):
            single, pair, rigids = self.trunk(single, pair, rigids, mask)
        
        noise_pred = self.noise_predictor(xt, timestep)

        return noise_t, noise_pred
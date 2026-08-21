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

    def forward(self, tokens, rigids, timestep, mask) -> tuple[Rigid, Rigid]:
        single, pair = self.input_embedder(tokens)
        rigids_t, noise_t = self.diffuser(x0=rigids, timestep=timestep, mask=mask)
        
        for _ in range(self.trunks):
            single, pair, rigids_t = self.trunk(single, pair, rigids=rigids_t, mask=mask)
        
        noise_pred = self.noise_predictor(rigids_t, timestep, mask=mask)

        return noise_t, noise_pred
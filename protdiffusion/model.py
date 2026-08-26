from torch import Tensor
import torch.nn as nn
from protdiffusion.modules import InputEmbedder, Trunk
from protdiffusion.diffusion import Diffuser, NoisePredictor, NoDiffusionPredictor
from protdiffusion.geometry import Rigid
from jaxtyping import Float, Int, Bool, jaxtyped
from beartype import beartype



class ProtDiffusion(nn.Module):
    def __init__(self, trunks=20, max_residues=2048):
        super().__init__()

        self.max_residues = max_residues
        self.input_embedder = InputEmbedder(self.max_residues)
        self.diffuser = Diffuser(num_timesteps=200)
        self.trunks = trunks
        self.trunk = Trunk()
        self.noise_predictor = NoisePredictor()

    @jaxtyped(typechecker=beartype)
    def forward(self, tokens: Int[Tensor, "B L" ], rigids: Rigid,
                 timestep: Int[Tensor, "B"], mask: Bool[Tensor, "B L"]) -> tuple[Rigid, Rigid]:
        single, pair = self.input_embedder(tokens)
        rigids_t, noise_t = self.diffuser(rigid_0=rigids, timestep=timestep, mask=mask)

        for _ in range(self.trunks):
            single, pair, rigids_t = self.trunk(single, pair, rigids=rigids_t, mask=mask)
        
        rigids_pred = self.noise_predictor(rigids_t, timestep)
        
        return rigids, rigids_pred #noise_t, noise_pred

    def __str__(self):
        return f"diffusion_t{self.trunks}"


class ProtNoDiffusion(nn.Module):
    def __init__(self, trunks, max_residues=2048):
        super().__init__()

        self.max_residues = max_residues
        self.input_embedder = InputEmbedder(self.max_residues)
        self.trunks = trunks
        self.trunk = Trunk()
        self.no_diffusion_predictor = NoDiffusionPredictor()

    @jaxtyped(typechecker=beartype)
    def forward(self, tokens: Int[Tensor, "B L"], rigids: Rigid, 
                mask: Bool[Tensor, "B L"]) -> tuple[Rigid, Rigid]:
        single, pair = self.input_embedder(tokens)
        
        for _ in range(self.trunks):
            single, pair, rigids = self.trunk(single, pair, rigids=rigids, mask=mask)
        
        rigids_pred = self.no_diffusion_predictor(rigids)

        return rigids, rigids_pred


    def __str__(self):
        return f"no_diffusion_t{self.trunks}"


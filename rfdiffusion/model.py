import torch 
import torch.nn as nn
from rfdiffusion.modules import InputEmbedder, Trunk
from rfdiffusion.diffusion import Diffuser


class RFDiffusion(nn.Module):
    def __init__(self, trunks):
        super().__init__()

        self.trunks = trunks
        self.input_embedder = InputEmbedder()
        self.trunk = Trunk()
        self.diffuser = Diffuser(timesteps=200)


    def forward(self, batch, mask):
        single, pair = self.input_embedder(batch)
        rigids = None
        for _ in range(self.trunks):
            single, pair, rigids = self.trunk(single, pair, rigids, mask)
        
        #xt, noise = self.diffuser(x0=rigids)
        return 
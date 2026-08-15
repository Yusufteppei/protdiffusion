import torch 
import torch.nn as nn
from rfdiffusion.modules import InputEmbedder, Trunk


class RFDiffusion(nn.Module):
    def __init__(self, trunks):
        super().__init__()

        self.trunks = trunks
        self.input_embedder = InputEmbedder()
        self.trunk = Trunk()


    def forward(self, batch, mask):
        single, pair = self.input_embedder(batch)
        rigids = None
        for _ in range(self.trunks):
            rigids = self.trunk(single, pair, rigids, mask)

        
        return single, pair, rigids
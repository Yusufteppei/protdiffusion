import torch 
import torch.nn as nn
from rfdiffusion.modules import InputEmbedder, Trunk


class RFDiffusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_embedder = InputEmbedder()
        self.trunk = Trunk()


    def forward(self, batch, mask):
        single, pair = self.input_embedder(batch)
        trunk = self.trunk(single, pair)

        
        
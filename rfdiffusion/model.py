import torch 
import torch.nn as nn
from rfdiffusion.modules import InputEmbedder


class RFDiffusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_embedder = InputEmbedder()


    def forward(self, batch, mask):
        single, pair = self.input_embedder(batch)
        print(single.shape, pair.shape)
        
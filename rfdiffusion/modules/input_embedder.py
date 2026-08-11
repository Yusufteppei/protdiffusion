import torch
import torch.nn as nn
from rfdiffusion.config import d_pair, d_res, n_res

class PositionEncoder(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):

        return x
    
class InputEmbedder(nn.Module):

    def __init__(self):
        super().__init__()

        self.res_emb = nn.Embedding(n_res, d_res)
        self.pos_encoder = PositionEncoder()

        self.left_proj = nn.Linear(d_res, d_pair)
        self.right_proj = nn.Linear(d_res, d_pair)


    def forward(self, batch):
        """
            Input is a seq sequence
        """
        single = self.res_emb(batch)
        single = self.pos_encoder(single)

        left = self.left_proj(single)
        right = self.right_proj(single)

        pair = (
            left[:,:,None,:]
            + right[:,None,:,:]
        )

        return single, pair
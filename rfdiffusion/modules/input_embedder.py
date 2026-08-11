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

        self.msa_emb = nn.Embedding(n_res, d_res)
        self.pos_encoder = PositionEncoder()

        self.left_proj = nn.Linear(d_res, d_pair)
        self.right_proj = nn.Linear(d_res, d_pair)


    def msa_process(self, msa):
        """
            Dummy function implementing a transform of the seq sequence embedding into
            a evolution-aware embedding
        """
        return msa

    def forward(self, batch):
        """
            Input is a seq sequence
        """
        msa = self.msa_emb(batch)
        msa = self.pos_encoder(msa)
        msa = self.msa_process(msa)

        left = self.left_proj(msa)
        right = self.right_proj(msa)

        pair = (
            left[:,:,None,:]
            + right[:,None,:,:]
        )

        return msa, pair
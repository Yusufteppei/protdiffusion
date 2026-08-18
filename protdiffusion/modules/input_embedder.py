import torch
import torch.nn as nn
from protdiffusion.config import d_pair, d_res, n_res
import math


class PositionEncoder(nn.Module):

    def __init__(self, max_len, d_enc):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_enc, 2) *
            (-math.log(10000.0) / d_enc)
        )

        pe = torch.zeros(max_len, d_enc)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        L = x.shape[1]

        return x + self.pe[:L]


class InputEmbedder(nn.Module):

    def __init__(self, max_residues):
        super().__init__()

        self.res_emb = nn.Embedding(n_res, d_res)
        self.pos_encoder = PositionEncoder(max_residues, d_res)

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
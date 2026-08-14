import torch
import torch.nn as nn
from rfdiffusion.config import d_pair, d_res, n_h
import math


class InvariantPointAttention(nn.Module):
    def __init__(self, num_heads=n_h, num_qk_points=4):
        super().__init__()

        assert d_res % num_heads == 0
        self.d_h = d_res // num_heads
        self.n_h = num_heads
        self.num_qk_points = num_qk_points

        self.qv_proj = nn.Linear(d_res, d_res) # d_res, d_h * n_h
        self.kv_proj = nn.Linear(d_res, d_res)
        self.vv_proj = nn.Linear(d_res, d_res)

        self.qp_proj = nn.Linear(d_res, num_heads * num_qk_points * 3) # 
        self.kp_proj = nn.Linear(d_res, num_heads * num_qk_points * 3)
        self.vp_proj = nn.Linear(d_res, num_heads * num_qk_points * 3)

        self.pair_bias_proj = nn.Linear(d_pair, n_h)
        
        self.o_proj = nn.Linear( 3*num_qk_points*num_heads + d_res, d_res)


    def forward(self, single, pair, rigids):
        B, L, d_res = single.shape

        Qp = self.qp_proj(single).view(B, L, n_h, self.num_qk_points, 3)
        Kp = self.kp_proj(single).view(B, L, n_h, self.num_qk_points, 3)
        Vp = self.vp_proj(single).view(B, L, n_h, self.num_qk_points, 3)

        
        Qp = rigids.apply(Qp).unsqueeze(2) # B, L, 1, H, qk, 3
        Kp = rigids.apply(Kp).unsqueeze(1) # B, 1, L, H, qk, 3
        Vp = rigids.apply(Vp).permute(0, 2, 1, 3, 4) # B, H, L, qk, 3
        Vp = Vp.flatten(-2) # B, H, L, 3*qk

 
        Qv = self.qv_proj(single).view(B, L, n_h, self.d_h).permute(0, 2, 1, 3) # B, H, L, d_h
        Kv = self.kv_proj(single).view(B, L, n_h, self.d_h).permute(0, 2, 1, 3)
        Vv = self.vv_proj(single).view(B, L, n_h, self.d_h).permute(0, 2, 1, 3)

        pair_score_bias = self.pair_bias_proj(pair) # B, L, L, H
        pair_score_bias = pair_score_bias.permute(0, 3, 1, 2)


        # Point Attention
        point_score = - ( Qp - Kp ).square().sum(dim=-1) / 2 # B, L, L, H, qk
        point_score = point_score.permute(0, 3, 1, 2, 4 ).sum(dim=-1)# B, H, L, L

        #print("Point score", point_score.shape)
    
        # Vector Attention
        vector_score =  torch.matmul( Qv, Kv.transpose(-2, -1) ) / math.sqrt(self.d_h) # B, H, L, L
        
        #print("Vector score", vector_score.shape)

        score_proj = torch.softmax( vector_score + point_score + pair_score_bias , dim=-1)


        vector_attention = score_proj@Vv # B, H, L, d_h
        vector_attention = vector_attention.permute(0, 2, 1, 3) # B, L, H, d_h
        vector_attention = vector_attention.flatten(-2) # B, L, d_res

        point_attention = score_proj@Vp # B, H, L, 3*qk
        point_attention = point_attention.permute(0, 2, 1, 3 ) # B, L, H, 3*qk
        point_attention = point_attention.flatten(-2) # B, L, H*3*qk

        attention = torch.concat([ vector_attention, point_attention ], dim=-1) # B, L, d_res + 3*qk*H

        print(f"Attention : {attention.shape}")
        out = self.o_proj(attention)
        print(f"IPA : {out.shape}")
        return out



class IPATransition(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_res, 4*d_res),
            nn.GELU(),
            nn.Linear(4*d_res, d_res)
        )


    def forward(self, single):
        single = self.net(single)

        return single

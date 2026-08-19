import torch
import torch.nn as nn
from protdiffusion.diffusion import TimeEmbedding
from protdiffusion.geometry import Rigid, Rotation
from protdiffusion.config import d_res


class NoisePredictor(nn.Module):

    def __init__(self):
        super().__init__()

        self.time_embedding = TimeEmbedding(128)
        self.translation_net = nn.Sequential(
            nn.Linear(6, 9),
            nn.GELU(),
            nn.Linear(9, 6),
            nn.GELU(),
            nn.Linear(6, 3)
        )
        self.rotation_net = nn.Sequential(
            nn.Linear(3, 6),
            nn.GELU(),
            nn.Linear(6, 3)
        )

    def forward(self, xt, timestep):

        rotation_time, translation_time = self.time_embedding(timestep)

        #rotation_noise = self.rotation_net(xt.rotation.matrix) + rotation_time[:, None, :])

        time = translation_time[:, None, :].expand(
            -1, xt.translation.shape[1], -1
        )
        x = torch.cat([xt.translation, time], dim=-1)
        

        print(x.shape)
        translation_noise = self.translation_net(x)

        noise = Rigid(rotation=Rotation.identity(), translation=translation_noise)

        return noise
import torch
import torch.nn as nn
from protdiffusiondiffusion import TimeEmbedding
from protdiffusiongeometry import Rigid, Rotation


class NoisePredictor(nn.Module):

    def __init__(self):
        super().__init__()

        self.time_embedding = TimeEmbedding(128)
        self.translation_net = nn.Sequential(
            nn.Linear(3, 6),
            nn.GELU(),
            nn.Linear(6, 3)
        )
        self.rotation_net = nn.Sequential(
            nn.Linear(3, 6),
            nn.GELU(),
            nn.Linear(6, 3)
        )

    def forward(self, xt, timestep):

        t = self.time_embedding(timestep)

        rotation_noise = self.rotation_net(xt.rotation.matrix + t[:, None, :])
        translation_noise = self.translation_net(xt.translation + t[:, None, :])

        noise = Rigid(rotation=Rotation(rotation_noise), translation=translation_noise)

        return noise
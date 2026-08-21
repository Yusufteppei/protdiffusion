import torch
import torch.nn as nn
from protdiffusion.diffusion import TimeEmbedding
from protdiffusion.geometry import Rigid, RotationVector


class NoisePredictor(nn.Module):

    def __init__(self):
        super().__init__()

        self.time_embedding = TimeEmbedding(128)

        self.translation_net = nn.Sequential(
            nn.Linear(6, 24),
            nn.GELU(),
            nn.Linear(24, 18),
            nn.GELU(),
            nn.Linear(18, 12),
            nn.GELU(),
            nn.Linear(12, 6),
            nn.GELU(),
            nn.Linear(6, 3)
        )

        self.rotation_net = nn.Sequential(
            nn.Linear(12, 48),
            nn.GELU(),
            nn.Linear(48, 36),
            nn.GELU(),
            nn.Linear(36, 24),
            nn.GELU(),
            nn.Linear(24, 18),
            nn.GELU(),
            nn.Linear(18, 12),
            nn.GELU(),
            nn.Linear(12, 6),
            nn.GELU(),
            nn.Linear(6, 3)
        )

    def forward(self, xt, timestep, mask=None) -> Rigid:

        rotation_time, translation_time = self.time_embedding(timestep)

        B, L = xt.translation.shape[:2]

        # -------------------------
        # Translation noise
        # -------------------------

        translation_time = translation_time[:, None, :].expand(
            -1, L, -1
        )

        translation_input = torch.cat(
            [xt.translation, translation_time],
            dim=-1
        )

        translation_noise = self.translation_net(
            translation_input
        )

        # -------------------------
        # Rotation noise
        # -------------------------

        rotation = xt.rotation.matrix.reshape(
            B, L, 9
        )

        rotation_time = rotation_time[:, None, :].expand(
            -1, L, -1
        )

        rotation_input = torch.cat(
            [rotation, rotation_time],
            dim=-1
        )

        rotation_noise = self.rotation_net(
            rotation_input
        )

        return Rigid(rotation_vector=RotationVector(rotation_noise), translation=translation_noise)
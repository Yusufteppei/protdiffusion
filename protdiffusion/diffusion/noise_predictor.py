import torch
import torch.nn as nn
from protdiffusion.diffusion import TimeEmbedding
from protdiffusion.geometry import Rigid, RotationVector


class NoisePredictor(nn.Module):

    def __init__(self):
        super().__init__()

        self.time_embedding = TimeEmbedding(128)

        self.translation_net = nn.Sequential(
            nn.Linear(131, 128),
            nn.GELU(),
            nn.Linear(128, 90),
            nn.GELU(),
            nn.Linear(90, 72),
            nn.GELU(),
            nn.Linear(72, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 24),
            nn.GELU(),
            nn.Linear(24, 12),
            nn.GELU(),
            nn.Linear(12, 6),
            nn.GELU(),
            nn.Linear(6, 3)
        )

        self.rotation_net = nn.Sequential(
            nn.Linear(131, 128),
            nn.GELU(),
            nn.Linear(128, 90),
            nn.GELU(),
            nn.Linear(90, 72),
            nn.GELU(),
            nn.Linear(72, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 24),
            nn.GELU(),
            nn.Linear(24, 12),
            nn.GELU(),
            nn.Linear(12, 6),
            nn.GELU(),
            nn.Linear(6, 3)
        )

    def forward(self, rigids_t, timestep) -> Rigid:

        rotation_time, translation_time = self.time_embedding(timestep)

        B, L = rigids_t.translation.shape[:2]

        # -------------------------
        # Translation noise
        # -------------------------

        translation_time = translation_time[:, None, :].expand(
            -1, L, -1
        )

        translation_input = torch.cat(
            [rigids_t.translation, translation_time],
            dim=-1
        )

        with open("logging", "w") as f:
            f.write(f"logs/translation_input: {translation_input.mean(dim=-1)} - {translation_input.std(dim=-1)}\n")
        translation_noise = self.translation_net(
            translation_input
        )

        # -------------------------
        # Rotation noise
        # -------------------------

        rotation = rigids_t.rotation.matrix.reshape(
            B, L, 9
        )
        rotation_vector = rigids_t.rotation_vector.vector

        rotation_time = rotation_time[:, None, :].expand(
            -1, L, -1
        )

        rotation_input = torch.cat(
            [rotation_vector, rotation_time],
            dim=-1
        )

        rotation_noise = self.rotation_net(
            rotation_input
        )

        return Rigid(rotation_vector=RotationVector(rotation_noise), translation=translation_noise)


class NoDiffusionPredictor(nn.Module):

    def __init__(self):
        super().__init__()

        self.translation_net = nn.Sequential(
            nn.Linear(3, 6),
            nn.GELU(),
            nn.Linear(6, 6),
            nn.GELU(),
            nn.Linear(6, 6),
            nn.GELU(),
            nn.Linear(6, 3)
        )

        self.rotation_net = nn.Sequential(
            nn.Linear(3, 6),
            nn.GELU(),
            nn.Linear(6, 6),
            nn.GELU(),
            nn.Linear(6, 6),
            nn.GELU(),
            nn.Linear(6, 3)
        )
        

    def forward(self, rigids) -> Rigid:
        B, L = rigids.translation.shape[:2]


        rotation = self.rotation_net(rigids.rotation_vector.vector)
        translation = self.translation_net(rigids.translation)

        return Rigid(rotation_vector=RotationVector(rotation), translation=translation)
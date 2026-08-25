from protdiffusion.geometry import Rigid, Rotation
import torch
from protdiffusion.config import ROOT_DIR


class RigidLoss:

    def __init__(self):
        pass


    def __call__(self, target1: Rigid, target2: Rigid, mask: torch.Tensor) -> torch.Tensor:

        translation_loss = torch.mean(torch.sum((target1.translation - target2.translation) ** 2, dim=-1) * mask)
        rotation_loss = torch.mean(torch.sum((target1.rotation.matrix - target2.rotation.matrix) ** 2,
                                              dim=(-1, -2)) * mask)

        with open(f"{ROOT_DIR}/logs/losses", "w") as f:
            f.write(f"Translation loss: {translation_loss}, Rotation loss: {rotation_loss}")
        return translation_loss + rotation_loss
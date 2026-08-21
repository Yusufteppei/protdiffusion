from protdiffusion.geometry import Rigid, Rotation
import torch


class RigidLoss:

    def __init__(self):
        pass


    def __call__(self, target1: Rigid, target2: Rigid, mask: torch.Tensor) -> torch.Tensor:

        translation_loss = torch.mean(torch.sum((target1.translation - target2.translation) ** 2, dim=-1) * mask)
        rotation_loss = torch.mean(torch.sum((target1.rotation.matrix - target2.rotation.matrix) ** 2,
                                              dim=(-1, -2)) * mask)

        print(f"rotation loss: {rotation_loss.item()}, translation loss: {translation_loss.item()}")

        return translation_loss# + rotation_loss 
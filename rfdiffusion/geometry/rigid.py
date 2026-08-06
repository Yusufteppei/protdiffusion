from __future__ import annotations
import torch
from rfdiffusion.geometry import Rotation


class Rigid:
    def __init__(self, rotation: Rotation, translation: torch.Tensor):
        self.rotation = rotation
        self.translation = translation


    def from_coords(cls, coords: torch.Tensor):

        return

    @classmethod
    def identity(cls):
        return cls(torch.eye(3))

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return self.rotation.apply(x) + self.translation


    def inverse(self) -> Rigid:
        rot_inv = self.rotation.inverse()
        t = -rot_inv.apply(self.translation)
        return Rigid(rot_inv, t)

    def compose(self, rigid2: Rigid) -> Rigid:
        """
            A.compose(B.compose(C)) = A x B x C
        """

        rot = self.rotation.compose(rigid2.rotation)
        trans = self.rotation.apply(rigid2.translation) + self.translation

        return Rigid(rotation=rot, translation=trans)
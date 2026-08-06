from __future__ import annotations
import torch


class Rotation:
    def __init__(self, matrix: torch.Tensor):
        self.matrix = matrix

    @classmethod
    def identity(cls):
        return cls(torch.eye(3))

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "...ij,...j->...i",
            self.matrix,
            x
        )

    def inverse(self) -> Rotation:
        return Rotation(self.matrix.transpose(-1, -2))

    def compose(self, rotation2: Rotation) -> Rotation:
        """
            A.compose(B.compose(C)) = A x B x C
        """

        return Rotation(torch.einsum(
            "...ij,...jk->...ik",
            self.matrix,
            rotation2.matrix
        ))



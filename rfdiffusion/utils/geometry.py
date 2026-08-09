import torch
import numpy as np


def normalize(x):
    
    norm = torch.linalg.norm(x, dim=-1).unsqueeze(-1)
    return x / norm


def dihedral(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor):
    """
        A dihedral A-B-C-D is defined as planes ABC and BCD

        Calculate the signed dihedral angle for four points A-B-C-D.

        Args:
            a, b, c, d: tensors of shape (..., 3)

        Returns:
            angle: tensor of shape (...,) in radians, range (-pi, pi]
    """

    b0 = b - a
    b1 = c - b
    b2 = d - c

    # Normalize the central bond
    b1 = b1 / torch.linalg.vector_norm(b1, dim=-1, keepdim=True)

    # Remove the component of b0 parallel to b1
    v = b0 - (b0 * b1).sum(dim=-1, keepdim=True) * b1

    # Remove the component of b2 parallel to b1
    w = b2 - (b2 * b1).sum(dim=-1, keepdim=True) * b1

    # Angle between the two projected vectors
    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1, v, dim=-1) * w).sum(dim=-1)

    return torch.atan2(y, x)


def build_atom(a, b, c, length: float, angle: float, torsion):
    pass
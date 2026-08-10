import torch
import numpy as np


def normalize(x):
    
    norm = torch.linalg.norm(x, dim=-1).unsqueeze(-1)
    return x / norm

import torch


def normalize(x: torch.Tensor):
    norm = torch.linalg.norm(x, dim=-1, keepdim=True)
    return x / norm


def dihedral(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
):
    """
    Signed dihedral angle for A-B-C-D.

    Inputs:
        (..., 3)

    Returns:
        (...,)
    """

    b0 = b - a
    b1 = c - b
    b2 = d - c

    b1 = normalize(b1)

    v = b0 - (b0 * b1).sum(dim=-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(dim=-1, keepdim=True) * b1

    x = (v * w).sum(dim=-1)

    y = (
        torch.cross(b1, v, dim=-1) * w
    ).sum(dim=-1)

    return torch.atan2(y, x)


def build_residue(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    length: torch.Tensor,
    bond_angle: torch.Tensor,
    torsion: torch.Tensor,
):
    """
    Construct D from A-B-C and internal coordinates.

    Inputs:
        a, b, c:      (..., 3)
        length:       (...)
        bond_angle:   (...)
        torsion:      (...)

    Returns:
        d:             (..., 3)
    """

    # C -> B direction
    e1 = normalize(c - b)

    # Normal to the ABC plane
    e2 = normalize(
        torch.cross(
            e1,
            c - a,
            dim=-1
        )
    )

    # In-plane direction perpendicular to e1
    e3 = torch.cross(e1, e2, dim=-1)

    direction = (
        -e1 * torch.cos(bond_angle).unsqueeze(-1)
        +
        torch.sin(bond_angle).unsqueeze(-1)
        *
        (
            e2 * torch.sin(torsion).unsqueeze(-1)
            +
            -e3 * torch.cos(torsion).unsqueeze(-1)
        )
    )

    d = c + length.unsqueeze(-1) * direction

    return d
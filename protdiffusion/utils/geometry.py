import torch
import numpy as np

import matplotlib.pyplot as plt
## CONSTANTS

# Bond lengths (Å)
N_CA = torch.tensor(1.4966)
CA_C = torch.tensor(1.5221)
C_N = torch.tensor(1.2991)
C_O = torch.tensor(1.231)

# Bond angles (degrees → radians)
CA_N_C = torch.deg2rad(torch.tensor(121.7))
N_CA_C = torch.deg2rad(torch.tensor(107.0100))
CA_C_N = torch.deg2rad(torch.tensor(116.2))
C_N_CA = torch.deg2rad(torch.tensor(118.9957))


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


def place_atom(
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


def build_residue_backbone(
    n: torch.Tensor,
    ca: torch.Tensor,
    c: torch.Tensor,
    phi: torch.Tensor,
    psi: torch.Tensor,
    omega: torch.Tensor,
):
    """
    Build the next residue's backbone.

    Inputs:
        n, ca, c : (..., 3)
            Backbone atoms of residue i.

        phi : (...)
            φ of residue i+1

        psi : (...)
            ψ of residue i

        omega : (...)
            ω peptide bond between i and i+1

    Returns:
        n_next, ca_next, c_next : (..., 3)
    """

    # ψ_i determines N_{i+1}
    n_next = place_atom(
        n,
        ca,
        c,       
        C_N,
        C_N_CA,
        psi,
    )

    # ω_i determines CA_{i+1}
    ca_next = place_atom(
        ca,
        c,
        n_next,
        N_CA,
        C_N_CA,
        omega,
    )

    # φ_{i+1} determines C_{i+1}
    c_next = place_atom(
        c,
        n_next,
        ca_next,
        CA_C,
        N_CA_C,
        phi,
    )

    return torch.stack([n_next, ca_next, c_next])




def plot_protein(rigids, mask=None, protein_idx=0):
    coords = rigids.translation[protein_idx].detach().cpu()

    if mask is not None:
        coords = coords[mask[protein_idx].cpu()]

    xyz = coords.numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
    )

    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()
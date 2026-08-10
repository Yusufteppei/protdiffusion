from rfdiffusion.data import load_pdb, get_coords
from inspect import signature
from Bio.PDB import Atom
from rfdiffusion.geometry import Rigid, Rotation
import torch
from rfdiffusion.data import Protein
from rfdiffusion.utils import build_residue_backbone
import math

DTYPE = torch.float32

a = load_pdb()

n = a.coords[0, 0]
ca = a.coords[0, 1]
c = a.coords[0, 2]

print(a.coords[:2])


phi = torch.tensor(math.radians(-60.), dtype=DTYPE)
psi = torch.tensor(math.radians(-45.), dtype=DTYPE)
omega = torch.tensor(math.pi, dtype=DTYPE)

next = build_residue_backbone(
    n, ca, c, phi, psi, omega
)

print("next", next)
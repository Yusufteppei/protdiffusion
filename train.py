from rfdiffusion.data import load_pdb, get_coords
from inspect import signature
from Bio.PDB import Atom
from rfdiffusion.geometry import Rigid, Rotation
import torch
from rfdiffusion.data import Protein

a = load_pdb()

R1 = torch.eye(3)
t1 = torch.tensor([10., 0., 0.])

A = Rigid(Rotation(R1), t1)

R2 = torch.eye(3)
t2 = torch.tensor([0., 5., 0.])

B = Rigid(Rotation(R2), t2)

C = A.compose(B)

point = torch.tensor([1., 2., 3.])

print(C.apply(point))
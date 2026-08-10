import torch
from rfdiffusion.utils import dihedral, place_atom
from rfdiffusion.data import Protein, load_pdb

p = load_pdb("datasets/1UBQ.pdb")

l1 = p.coords[0]
l2 = p.coords[1]

torsion = dihedral(*l1, l2[0])
length = torch.Tensor([1.2991])
bond_angle = torch.tensor(torch.deg2rad(torch.tensor(114.2044)))
print(f"Derived Torsion : {torsion}")

placement = place_atom(*l1, length, bond_angle, torsion)
print(f"Target Placement: {l2[0]}")
print(f"Derived Placement: {placement}")

target = l2[0]

actual_length = torch.linalg.vector_norm(target - l1[2])

print("Actual C-N length:", actual_length)

def angle(a, b, c):
    ba = a - b
    bc = c - b

    ba = ba / torch.linalg.norm(ba)
    bc = bc / torch.linalg.norm(bc)

    return torch.acos((ba * bc).sum(-1))

bond_angle = angle(l1[1], l1[2], l2[0])

print("Actual angle:", torch.rad2deg(bond_angle))


import torch
from rfdiffusion.utils import dihedral, build_residue
a = torch.tensor([0.0, 1.0, 0.0])
b = torch.tensor([0.0, 0.0, 0.0])
c = torch.tensor([1.0, 0.0, 0.0])

length = torch.tensor(1.0)
bond_angle = torch.tensor(torch.pi / 2)
torsion = torch.tensor(torch.pi / 3)

d = build_residue(
    a,
    b,
    c,
    length,
    bond_angle,
    torsion,
)

print("D:", d)

phi = dihedral(a, b, c, d)

print("Target torsion:", torsion)
print("Recovered torsion:", phi)
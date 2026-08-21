import math
import torch
from protdiffusion.data import Protein
from protdiffusion.utils import (
    dihedral, place_atom, build_residue_backbone
)
from protdiffusion.utils.geometry import (
    C_N,
    C_N_CA,
    N_CA,
    N_CA_C,
    CA_C,
)
    
load_pdb = Protein.load_pdb
DTYPE = torch.float32
ATOL = 1e-5


def angle(a, b, c):
    """Return angle ABC in radians."""
    ba = a - b
    bc = c - b
    ba = ba / torch.linalg.norm(ba, dim=-1, keepdim=True)
    bc = bc / torch.linalg.norm(bc, dim=-1, keepdim=True)
    cosine = (ba * bc).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.acos(cosine)


def angle_difference(a, b):
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


def test_place_atom_round_trip():
    a = torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE)
    b = torch.tensor([0.0, 0.0, 0.0], dtype=DTYPE)
    c = torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE)

    length = torch.tensor(1.33, dtype=DTYPE)
    bond_angle = torch.tensor(math.radians(121.0), dtype=DTYPE)
    torsion = torch.tensor(math.radians(60.0), dtype=DTYPE)

    d = place_atom(a, b, c, length, bond_angle, torsion)

    assert torch.allclose(torch.linalg.norm(d - c), length, atol=ATOL)
    assert torch.allclose(angle(b, c, d), bond_angle, atol=ATOL)
    assert torch.allclose(dihedral(a, b, c, d), torsion, atol=ATOL)


def test_place_atom_batch():
    a = torch.tensor([[0., 1., 0.], [0., 2., 0.]], dtype=DTYPE)
    b = torch.tensor([[0., 0., 0.], [0., 0., 0.]], dtype=DTYPE)
    c = torch.tensor([[1., 0., 0.], [2., 0., 0.]], dtype=DTYPE)

    length = torch.tensor([1.33, 1.40], dtype=DTYPE)
    bond_angle = torch.tensor(
        [math.radians(121.), math.radians(118.)], dtype=DTYPE
    )
    torsion = torch.tensor(
        [math.radians(60.), math.radians(-45.)], dtype=DTYPE
    )

    d = place_atom(a, b, c, length, bond_angle, torsion)

    assert d.shape == (2, 3)
    recovered = torch.stack(
        [dihedral(a[i], b[i], c[i], d[i]) for i in range(2)]
    )
    assert torch.allclose(recovered, torsion, atol=ATOL)


def test_backbone_bond_lengths():
    a = load_pdb()

    n = a.coords[0, 0]
    ca = a.coords[0, 1]
    c = a.coords[0, 2]

    phi = torch.tensor(math.radians(-60.), dtype=DTYPE)
    psi = torch.tensor(math.radians(-45.), dtype=DTYPE)
    omega = torch.tensor(math.pi, dtype=DTYPE)

    next = build_residue_backbone(
        n, ca, c, phi, psi, omega
    )
    n_next = next[0]
    ca_next = next[1]
    c_next = next[2]

    print(f"next: {next}")
    assert torch.allclose(torch.linalg.norm(n_next - c), C_N, atol=ATOL)
    assert torch.allclose(torch.linalg.norm(ca_next - n_next), N_CA, atol=ATOL)
    assert torch.allclose(torch.linalg.norm(c_next - ca_next), CA_C, atol=ATOL)


def test_backbone_bond_angles():
    a = load_pdb()

    n = a.coords[0, 0]
    ca = a.coords[0, 1]
    c = a.coords[0, 2]

    phi = torch.tensor(math.radians(-60.), dtype=DTYPE)
    psi = torch.tensor(math.radians(-45.), dtype=DTYPE)
    omega = torch.tensor(math.pi, dtype=DTYPE)

    next = build_residue_backbone(
            n, ca, c, phi, psi, omega
    )
    n_next = next[0]
    ca_next = next[1]
    c_next = next[2]

    assert torch.allclose(angle(n, ca, c), N_CA_C, atol=ATOL)
    assert torch.allclose(angle(c, n_next, ca_next), C_N_CA, atol=ATOL)
    assert torch.allclose(angle(n_next, ca_next, c_next), N_CA_C, atol=ATOL)


def test_backbone_torsions():
    a = load_pdb()

    n = a.coords[0, 0]
    ca = a.coords[0, 1]
    c = a.coords[0, 2]


    phi = torch.tensor(math.radians(-60.), dtype=DTYPE)
    psi = torch.tensor(math.radians(-45.), dtype=DTYPE)
    omega = torch.tensor(math.pi, dtype=DTYPE)

    next = build_residue_backbone(
        n, ca, c, phi, psi, omega
    )
    n_next = next[0]
    ca_next = next[1]
    c_next = next[2]

    #ATOL = 

    assert torch.allclose(
        angle_difference(
            dihedral(ca, c, n_next, ca_next),
            omega,
        ),
        torch.tensor(0.0),
        atol=ATOL,
    )
    assert torch.allclose(
            angle_difference(
                dihedral(n, ca, c, n_next),
                psi,
            ),
            torch.tensor(0.0),
            atol=ATOL,
        )
    assert torch.allclose(
            angle_difference(
                dihedral(c, n_next, ca_next, c_next),
                phi,
            ),
            torch.tensor(0.0),
            atol=ATOL,
    )


def test_backbone_reconstruction_from_pdb():
    p = load_pdb("datasets/1UBQ.pdb")

    residue_i = p.coords[0]
    residue_next = p.coords[1]

    n = residue_i[0]
    ca = residue_i[1]
    c = residue_i[2]

    n_next = residue_next[0]
    ca_next = residue_next[1]
    c_next = residue_next[2]

    phi = dihedral(c, n_next, ca_next, c_next)
    psi = dihedral(n, ca, c, n_next)
    omega = dihedral(ca, c, n_next, ca_next)

    pred_n, pred_ca, pred_c = build_residue_backbone(
        n, ca, c,
        phi, psi, omega
    )


    ATOL = 0.2

    assert torch.allclose(pred_n, n_next, atol=ATOL)
    assert torch.allclose(pred_ca, ca_next, atol=ATOL)
    assert torch.allclose(pred_c, c_next, atol=ATOL)


    
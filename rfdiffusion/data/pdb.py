from Bio.PDB import PDBParser
from rfdiffusion.data.protein import Protein
from Bio.PDB.Structure import Structure
import torch
from Bio.SeqUtils import seq1


def get_sequence(structure: Structure) -> str:
    """
    Extract the protein sequence from a Bio.PDB Structure.

    Non-standard residues, waters, ligands, and hetero residues
    are ignored.

    Returns:
        str: one-letter amino-acid sequence
    """
    sequence = []

    for residue in structure.get_residues():
        # ' ' means a standard ATOM residue rather than HETATM
        if residue.id[0] != ' ':
            continue

        sequence.append(seq1(residue.resname))

    return ''.join(sequence)

parser = PDBParser(QUIET=True)


def get_coords(pdb_object: Structure) -> torch.Tensor:
    p = pdb_object

    coords = torch.Tensor()
    residues =  p.get_residues()
    while True:
        try:
            res = next(residues)
            if res.id[0] != ' ':
                continue
            #print(res)
        except:
            break
        atoms = res.get_atoms()

        N = next(atoms)
        #print(N)
        assert N.name == 'N'

        CA = next(atoms)
        #print(CA)
        assert CA.name == 'CA'

        C = next(atoms)
        #print(C)
        assert C.name == 'C'

        N = torch.Tensor(N.coord).unsqueeze(0)
        CA = torch.Tensor(CA.coord).unsqueeze(0)
        C = torch.Tensor(C.coord).unsqueeze(0)
        
        coord = torch.concatenate([N, CA, C]).unsqueeze(0)
        #print("COORD - NCAC", coord.shape)
        coords = torch.concatenate([coords, coord])
    #print("COORDS", coords.shape)
    return coords



def load_pdb(path="datasets/1UBQ.pdb")-> Protein:
    structure = parser.get_structure("protein", path)
    seq = get_sequence(structure)
    coords = get_coords(structure)
    protein = Protein.from_coords(coords, sequence=seq)
    return protein


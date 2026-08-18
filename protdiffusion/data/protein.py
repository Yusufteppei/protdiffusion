from __future__ import annotations
import torch
from protdiffusion.geometry import Rigid
import shutil
from Bio.PDB import PDBParser, PDBList
from Bio.PDB.Structure import Structure
from Bio.SeqUtils import seq1

pdbl = PDBList()
parser = PDBParser(QUIET=True)


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


def get_coords(pdb_object: Structure) -> torch.Tensor:
    # FIX BACKBONE ORDER RELIANCE TO ACCOMODATE NON-PROTEIN RESIDUES
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


class Protein:
    coords: torch.Tensor      # (..., L, atoms, 3)
    rigids: Rigid             # (..., L)
    sequence: torch.Tensor    # (..., L)
    mask: torch.BoolTensor    # (..., L)


    def __init__(self, 
                 rigids: Rigid, 
                 sequence: str | None = None, 
                 mask: torch.Tensor | None = None, 
                 coords: torch.Tensor | None = None,
                 seq_tokens: torch.Tensor | None = None,
                ):
        """
          coords: In case PDB coordinates are passed in inference instead
                  of having to convert to rigids manually
        """
        self.rigids = rigids
        self.sequence = sequence
        self.mask = mask
        self.coords = coords
        self.pad_rigid = Rigid.identity()

        if sequence is not None:
            rigids.translation.shape[0] == len(sequence)

    def __str__(self):
        return f"Protein <{self.__len__()}>"
    
    @classmethod
    def load_pdb(cls, code="1UBQ")-> Protein:
        ## Deprecate the path
        try:
            structure = parser.get_structure("protein", f"proteins/{code}.pdb")
        except:
            path = pdbl.retrieve_pdb_file(code, pdir="proteins", file_format="pdb", obsolete=False)
            shutil.move(path, f"proteins/{code}.pdb")
            
            structure = parser.get_structure("protein", f"proteins/{code}.pdb")

        #seq = get_sequence(structure)
        #coords = get_coords(structure)
        #protein = Protein.from_coords(coords, sequence=seq)
        return structure

    @classmethod
    def from_coords(cls, coords: torch.Tensor, sequence=None, mask=None, seq_tokens=None):
        
        rigids = Rigid.from_coords(coords)
        protein = cls(rigids=rigids, coords=coords, sequence=sequence, mask=mask, seq_tokens=seq_tokens)

        protein.coords = coords
        return protein

    @classmethod
    def from_code(cls, code: str="1UBQ", mask=None, seq_tokens=None):
        obj = cls.load_pdb(code)
        coords = get_coords(obj)
        sequence = get_sequence(obj)
        rigids = Rigid.from_coords(coords=coords)
        protein = cls(rigids=rigids, coords=coords, sequence=sequence, mask=mask, seq_tokens=seq_tokens)

        return protein

    @classmethod
    def from_codes(cls, codes: list):
        return [ cls.from_code(i) for i in codes]

    def __len__(self):
        return self.rigids.translation.shape[0]

    def extend(self, max_len):
        self.rigids += [self.pad_rigid] * (max_len - len(self.rigids))

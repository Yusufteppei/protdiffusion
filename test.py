import torch
from protdiffusion.utils import dihedral, place_atom
from protdiffusion.data import Protein
from protdiffusion.model import ProtDiffusion

model = ProtDiffusion()

protein = Protein.from_code("1UBQ")

rigids = protein.rigids

rigids.to_pdb("1UBQ")
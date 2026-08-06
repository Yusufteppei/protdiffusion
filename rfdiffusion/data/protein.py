import torch
from rfdiffusion.geometry import Rigid


class Protein:
    coords: torch.Tensor      # (..., L, atoms, 3)
    rigids: Rigid             # (..., L)
    sequence: torch.Tensor    # (..., L)
    mask: torch.BoolTensor    # (..., L)

    assert rigids.translation.shape[-2] == sequence.shape[-1]

    def __init__(self, rigids, sequence=None, mask=None, coords=None):
        """
          coords: In case PDB coordinates are passed in inference instead
                  of having to convert to rigids manually
        """
        self.rigids = rigids
        self.sequence = sequence
        self.mask = mask
        self.coords = coords

    @classmethod
    def from_coords(cls, coords):
        rigids = Rigid.rigid_from_coord(coords)
        protein = cls(rigids=rigids)

        return protein


    def __len__(self):
        return self.rigids.rotation.shape[-3]


    def coords_to_rigids(self, coords: torch.Tensor) -> Rigid:
        pass
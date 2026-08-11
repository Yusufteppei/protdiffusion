import torch
from rfdiffusion.geometry import Rigid



class Protein:
    coords: torch.Tensor      # (..., L, atoms, 3)
    rigids: Rigid             # (..., L)
    sequence: torch.Tensor    # (..., L)
    mask: torch.BoolTensor    # (..., L)


    def __init__(self, 
                 rigids: list[Rigid], 
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

        if sequence is not None:
            len(rigids) == len(sequence)

    def __str__(self):
        return f"Protein <{self.__len__()}>"

    @classmethod
    def from_coords(cls, coords: torch.Tensor, sequence=None, mask=None, seq_tokens=None):
        
        rigids = Rigid.from_coords(coords)
        protein = cls(rigids=rigids, coords=coords, sequence=sequence, mask=mask, seq_tokens=seq_tokens)

        protein.coords = coords
        return protein


    def __len__(self):
        return len(self.rigids)

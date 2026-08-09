from __future__ import annotations
import torch
from rfdiffusion.utils import normalize


class Rotation:
    def __init__(self, matrix: torch.Tensor):
        self.matrix = matrix

    @classmethod
    def identity(cls):
        return cls(torch.eye(3))

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "...ij,j->i",
            self.matrix,
            x
        )

    def inverse(self) -> Rotation:
        return Rotation(self.matrix.transpose(-1, -2))

    def compose(self, rotation2: Rotation) -> Rotation:
        """
            A.compose(B.compose(C)) = A x B x C
        """

        return Rotation(torch.einsum(
            "...ij,...jk->...ik",
            self.matrix,
            rotation2.matrix
        ))


class Rigid:
    def __init__(self, rotation: Rotation, translation: torch.Tensor):
        self.rotation = rotation
        self.translation = translation


    @classmethod
    def from_coords(cls, coords: torch.Tensor):
        """
            ORDER - N, CA, C
            C - CA : x axis
        """
        #print(coords.shape)
        #N = coords[..., 0] # <-----------------
        #CA = coords[..., 1] # <-----------------
        #C = coords[..., 2] # <-----------------

        #a = C - CA
        #b = N - CA

        CA = coords[:,1,:].unsqueeze(1).expand(-1, 3, 3)
        #print("CA", CA.shape)

        a = ( coords - CA ) [:,2]
        b = ( coords - CA ) [:,0]

        a_ = normalize(a)
        b_ = normalize(b)

        e1 = a_ 

        x = torch.einsum("bi,bj->b",b_, e1)
        #print(x)
        projection = torch.sum(b * e1, dim=-1, keepdim=True)
        e2 = normalize(b - projection * e1)
        e3 = torch.cross(e1, e2, dim=-1)

        R = torch.stack([e1, e2, e3], dim=-1)
        t = CA[:, 1,:]
        #print(f"rigids shape: {R.shape}")
        rigids = [ cls(rotation=Rotation(R[i]), translation=t[i]) for i in range(R.shape[-3]) ]
        return rigids



    @classmethod
    def identity(cls):
        return cls(torch.eye(3))

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return self.rotation.apply(x) + self.translation


    def inverse(self) -> Rigid:
        rot_inv = self.rotation.inverse()
        t = -rot_inv.apply(self.translation)
        return Rigid(rot_inv, t)

    def compose(self, rigid2: Rigid) -> Rigid:
        """
            A.compose(B.compose(C)) = A x B x C
        """

        rot = self.rotation.compose(rigid2.rotation)
        trans = self.rotation.apply(rigid2.translation) + self.translation

        return Rigid(rotation=rot, translation=trans)
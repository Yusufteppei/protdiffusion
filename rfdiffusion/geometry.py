from __future__ import annotations
import torch
from rfdiffusion.utils import normalize


class Rotation:
    def __init__(self, matrix: torch.Tensor):
        self.matrix = matrix

    @staticmethod
    def from_rotvec(rotvec, eps=1e-8):
        """
        Convert rotation vectors to rotation matrices using
        the SO(3) exponential map.

        Args:
            rotvec: (..., 3)
            eps: numerical stability constant

        Returns:
            Rotation with rotation matrix (..., 3, 3)
        """
        theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True)

        # Skew-symmetric matrix [w]_x
        x, y, z = rotvec.unbind(dim=-1)

        zero = torch.zeros_like(x)

        K = torch.stack([
            zero, -z,    y,
            z,     zero, -x,
            -y,    x,    zero
        ], dim=-1).reshape(*rotvec.shape[:-1], 3, 3)

        # Rodrigues:
        #
        # R = I + sin(theta)/theta K
        #       + (1-cos(theta))/theta² K²

        theta2 = theta.square()

        A = torch.where(
            theta2 > eps,
            torch.sin(theta) / theta,
            1.0 - theta2 / 6.0
        )

        B = torch.where(
            theta2 > eps,
            (1.0 - torch.cos(theta)) / theta2,
            0.5 - theta2 / 24.0
        )

        I = torch.eye(
            3,
            device=rotvec.device,
            dtype=rotvec.dtype
        )

        R = (
            I
            + A[..., None] * K
            + B[..., None] * (K @ K)
        )

        return Rotation(R)

    @classmethod
    def identity(cls):
        return cls(torch.eye(3))

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        R = self.matrix
        while R.ndim < x.ndim + 1:
            R = R.unsqueeze(-3)
        return torch.einsum(
            "...ij,...j->...i",
            R,
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
        

        CA = coords[:,1,:].unsqueeze(1).expand(-1, 3, 3)

        a = ( coords - CA ) [:,2]
        b = ( coords - CA ) [:,0]

        a_ = normalize(a)
        b_ = normalize(b)

        e1 = a_ 

        
        projection = torch.sum(b * e1, dim=-1, keepdim=True)
        e2 = normalize(b - projection * e1)
        e3 = torch.cross(e1, e2, dim=-1)

        R = torch.stack([e1, e2, e3], dim=-1)
        t = CA[:, 1,:]
        rigids = [ cls(rotation=Rotation(R[i]), translation=t[i]) for i in range(R.shape[-3]) ]
        return rigids



    @classmethod
    def identity(cls):
        return cls(torch.eye(3))

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        R = self.rotation.apply(x)
        t = self.translation
        while t.ndim < R.ndim:
            t = t.unsqueeze(-2)
        return R + t


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


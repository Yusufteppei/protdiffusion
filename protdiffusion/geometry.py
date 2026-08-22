from __future__ import annotations
import torch
from jaxtyping import jaxtyped, Float
from protdiffusion.utils import normalize
from beartype import beartype


class RotationVector:
    def __init__(self, vector: Float[torch.Tensor, "B L 3"]):
        self.vector = vector

    @classmethod
    def from_rotation(cls, rotation: Rotation):
        """
        Convert a rotation matrix to a rotation vector using
        the SO(3) logarithm map.

        Args:
            rotation: (..., 3, 3)

        Returns:
            RotationVector with vector (..., 3)
        """
        R = rotation.matrix
        theta = torch.acos(torch.clamp((torch.diagonal(R, dim1=-2, dim2=-1).sum(-1) - 1) / 2, -1 + 1e-7, 1 - 1e-7))
        sin_theta = torch.sin(theta)

        # Avoid division by zero for small angles
        small_angle_mask = sin_theta.abs() < 1e-7
        sin_theta[small_angle_mask] = 1.0

        log_R = (theta / (2 * sin_theta))[..., None, None] * (R - R.transpose(-1, -2))
        rotvec = torch.stack([log_R[..., 2, 1], log_R[..., 0, 2], log_R[..., 1, 0]], dim=-1)

        return cls(rotvec)

    
class Rotation:
    @jaxtyped(typechecker=beartype)
    def __init__(self, matrix: Float[torch.Tensor, "... 3 3"]):
        self.matrix = matrix

    
    @staticmethod
    @jaxtyped(typechecker=beartype)
    def from_rotvec(rotvec: Float[torch.Tensor, "B L 3"], eps=1e-8):
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


    @jaxtyped(typechecker=beartype)
    def apply(self, x: Float[torch.Tensor, "... 3"]) -> Float[torch.Tensor, "... 3"]:
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
    @jaxtyped(typechecker=beartype)
    def __init__(self, rotation: Rotation = None, translation: Float[torch.Tensor, "... 3"] = None, 
                 rotation_vector: RotationVector = None):
        if rotation is None and rotation_vector is not None:
            rotation = Rotation.from_rotvec(rotation_vector.vector)
            #print("derived rotation shape", rotation.matrix.shape)
        self.rotation = rotation

        self.translation = translation

        if rotation_vector is None and rotation is not None:
            rotation_vector = RotationVector.from_rotation(rotation)
        self.rotation_vector = rotation_vector 


    
    @classmethod
    @jaxtyped(typechecker=beartype)
    def from_coords(cls, coords: Float[torch.Tensor, "N 3 3"]):
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
        rigids = cls.from_list(rigids)
        return rigids



    @classmethod
    def identity(cls):
        return cls(Rotation.identity(), torch.zeros(3))

    @jaxtyped(typechecker=beartype)
    def apply(self, x: Float[torch.Tensor, "... 3"]) -> Float[torch.Tensor, "... 3"]:
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


    @classmethod
    def from_list(cls, rigids_list: list[Rigid]):
        max_len = max([ r.translation.shape[0] for r in rigids_list])
        translations = torch.stack([ r.translation for r in rigids_list])
        rotations = torch.stack([ r.rotation.matrix for r in rigids_list])

        new_rigid = cls(rotation=Rotation(rotations), translation=translations)

        return new_rigid


    def to_list(self) -> list[Rigid]:
        return [ Rigid(Rotation(self.rotation.matrix[i]), self.translation[i]) for i in range(self.translation.shape[0]) ]


    def _extend(self, max_len):
        len_ = self.translation.shape[0]
        gap = max_len - len_

        if gap > 0:
            pad_rigid = Rigid.identity()
            pad_rigids = [pad_rigid] * gap
            rigids_list = self.to_list() + pad_rigids
            return Rigid.from_list(rigids_list)
        else:
            return self
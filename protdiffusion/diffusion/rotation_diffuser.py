from torch import linspace, Tensor, zeros, ones, linspace, matmul, randn, matrix_exp
from jaxtyping import Float, jaxtyped, Bool, Int
from beartype import beartype
import torch.nn as nn
from protdiffusion.geometry import Rotation, RotationVector


class RotationDiffuser(nn.Module):

    def __init__(self, num_timesteps, sigma_start=1e-4, sigma_end=1.0):

        super().__init__()

        sigma = linspace(sigma_start, sigma_end, num_timesteps)

        self.register_buffer("sigma", sigma)


    @jaxtyped(typechecker=beartype)
    def forward(self, R0: Rotation, timestep: Int[Tensor, "B"], 
                noise: Float[Tensor, "B L 3"] = None, 
                mask: Bool[Tensor, "B L"] = None) -> tuple[Rotation, RotationVector]:
        """
        R0:       (..., 3, 3)
        timestep: (B,)

        noise:    (..., 3)
                  Rotation vectors in the Lie algebra so(3)
        """

        if noise is None:
            noise = randn(
                *R0.matrix.shape[:-2],
                3,
                device=R0.matrix.device,
                dtype=R0.matrix.dtype,
            )

        if mask is None:
            mask = ones(
                *R0.matrix.shape[:-2],
                device=R0.matrix.device,
                dtype=torch.bool,
            )
            
        sigma = self.sigma[timestep]

        while sigma.ndim < noise.ndim:
            sigma = sigma.unsqueeze(-1)

        # Scale the rotation-vector noise according to the timestep.
        omega = sigma * noise

        # Convert the rotation vector into a skew-symmetric matrix.
        K = zeros(
            *omega.shape[:-1],
            3,
            3,
            device=omega.device,
            dtype=omega.dtype,
        )

        K[..., 0, 1] = -omega[..., 2]
        K[..., 0, 2] =  omega[..., 1]
        K[..., 1, 0] =  omega[..., 2]
        K[..., 1, 2] = -omega[..., 0]
        K[..., 2, 0] = -omega[..., 1]
        K[..., 2, 1] =  omega[..., 0]

        # Exponentiate the skew-symmetric matrix to obtain
        # a valid rotation matrix in SO(3).
        R_noise = matrix_exp(K)

        # Compose the noise rotation with the original rotation.
        Rt = matmul(R_noise, R0.matrix) * mask.unsqueeze(-1).unsqueeze(-1)

        noise = noise * mask.unsqueeze(-1)
        
        return Rotation(Rt), RotationVector(noise)
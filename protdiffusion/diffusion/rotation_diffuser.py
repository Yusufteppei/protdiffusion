import torch
import torch.nn as nn
from protdiffusion.geometry import Rotation, RotationVector


class RotationDiffuser(nn.Module):

    def __init__(self, num_timesteps, sigma_start=1e-4, sigma_end=1.0):

        super().__init__()

        sigma = torch.linspace(sigma_start, sigma_end, num_timesteps)

        self.register_buffer("sigma", sigma)


    def forward(self, R0: Rotation, timestep: int, 
                noise: torch.Tensor =None) -> tuple[Rotation, RotationVector]:
        """
        R0:       (..., 3, 3)
        timestep: (B,)

        noise:    (..., 3)
                  Rotation vectors in the Lie algebra so(3)
        """

        if noise is None:
            noise = torch.randn(
                *R0.matrix.shape[:-2],
                3,
                device=R0.matrix.device,
                dtype=R0.matrix.dtype,
            )

        sigma = self.sigma[timestep]

        while sigma.ndim < noise.ndim:
            sigma = sigma.unsqueeze(-1)

        # Scale the rotation-vector noise according to the timestep.
        omega = sigma * noise

        # Convert the rotation vector into a skew-symmetric matrix.
        K = torch.zeros(
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
        R_noise = torch.matrix_exp(K)

        # Compose the noise rotation with the original rotation.
        Rt = torch.matmul(R_noise, R0.matrix)
        
        
        return Rotation(Rt), RotationVector(noise)
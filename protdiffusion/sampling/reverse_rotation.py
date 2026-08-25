import torch 
import torch.nn as nn
from jaxtyping import Float, jaxtyped
from beartype import beartype



class ReverseRotation(nn.Module):
    def __init__(self):
        super().__init__()

    @jaxtyped(typechecker=beartype)
    def forward(self, rotation: Float[torch.Tensor, "... 3 3"]) -> Float[torch.Tensor, "... 3 3"]:
        """
        Args:
            rotation: Tensor of shape (B, L, 3, 3) representing rotation matrices.
        Returns:
            Tensor of shape (B, L, 3, 3) representing the inverse of the input rotation matrices.
        """
        return rotation.transpose(-1, -2)
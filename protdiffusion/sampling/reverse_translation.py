import torch 
import torch.nn as nn
from jaxtyping import Float, jaxtyped
from beartype import beartype



class ReverseTranslation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, translation: Float[torch.Tensor,  "... 3"]) -> Float[torch.Tensor, "... 3"]:
        """
        Args:
            translation: Tensor of shape (B, L, 3) representing translation vectors.
        Returns:
            Tensor of shape (B, L, 3) representing the inverse of the input translation vectors.
        """
        return -translation
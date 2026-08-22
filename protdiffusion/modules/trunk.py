from torch import zeros, Tensor
from jaxtyping import Float, jaxtyped, Bool
from beartype import beartype
import torch.nn as nn
from protdiffusion.modules import InvariantPointAttention, IPATransition, BackboneUpdate
from protdiffusion.geometry import Rigid, Rotation
from protdiffusion.config import d_res


class Trunk(nn.Module):
    def __init__(self):
        super().__init__()

        self.ipa = InvariantPointAttention()
        self.layer_norm = nn.LayerNorm(d_res)
        self.ipa_transition = IPATransition()
        self.backbone_update = BackboneUpdate()


    @jaxtyped(typechecker=beartype)
    def forward(self, single: Float[Tensor, "B L d_res"], pair: Float[Tensor, "B L L d_pair"], 
                rigids: Rigid = None, mask: Bool[Tensor, "B L"] = None):
        
        single = single * mask[..., None]
        pair = pair * mask[..., None, None]
        B, L, _ = single.shape
        if rigids is None:
            rigids = Rigid(
                Rotation.identity(),
                zeros(B, L, 3)
            )

        single = single + self.ipa(single, pair, rigids, mask)
        single = self.layer_norm(single)
        single = single + self.ipa_transition(single)

        rot_lin, translation = self.backbone_update(single)

        rotation = Rotation.from_rotvec(rot_lin)

        rigids = Rigid(rotation, translation).compose(rigids)

        return single, pair, rigids
import torch
import torch.nn as nn
from rfdiffusion.modules import InvariantPointAttention, IPATransition, BackboneUpdate
from rfdiffusion.geometry import Rigid, Rotation
from rfdiffusion.config import d_res


class Trunk(nn.Module):
    def __init__(self):
        super().__init__()

        self.ipa = InvariantPointAttention()
        self.layer_norm = nn.LayerNorm(d_res)
        self.ipa_transition = IPATransition()
        self.backbone_update = BackboneUpdate()


    def forward(self, single, pair, rigids=None):

        B, L, _ = single.shape
        if rigids is None:
            rigids = Rigid(
                Rotation.identity(),
                torch.zeros(B, L, 3)
            )

        single = single + self.ipa(single, pair, rigids)
        single = self.layer_norm(single)
        single = single + self.ipa_transition(single)

        rot_lin, translation = self.backbone_update(single)

        rotation = Rotation.from_rotvec(rot_lin)

        rigids = Rigid(rotation, translation).compose(rigids)

        return rigids
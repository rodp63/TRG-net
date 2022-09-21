from torch import nn

from trgnet.backbones.fpn import FeaturePyramidNetwork, LastLevelMaxPool
from trgnet.utils import IntermediateLayerGetter


class BackboneWithFPN(nn.Module):
    def __init__(
        self,
        backbone,
        return_layers,
        in_channels_list,
        out_channels,
        extra_blocks,
    ):
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


def mobilenet_extractor(
    backbone,
    fpn,
    trainable_layers,
    returned_layers=None,
    extra_blocks=None,
):
    backbone = backbone.features
    stage_indices = (
        [0]
        + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)]
        + [len(backbone) - 1]
    )
    num_stages = len(stage_indices)

    if trainable_layers < 0 or trainable_layers > num_stages:
        raise ValueError(
            f"Trainable layers should be in the range [0,{num_stages}], got {trainable_layers}"
        )
    freeze_before = (
        len(backbone)
        if trainable_layers == 0
        else stage_indices[num_stages - trainable_layers]
    )

    for b in backbone[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)

    out_channels = 256
    if fpn:
        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()
        if returned_layers is None:
            returned_layers = [num_stages - 2, num_stages - 1]
        return_layers = {
            f"{stage_indices[k]}": str(v) for v, k in enumerate(returned_layers)
        }
        in_channels_list = [
            backbone[stage_indices[i]].out_channels for i in returned_layers
        ]
        return BackboneWithFPN(
            backbone,
            return_layers,
            in_channels_list,
            out_channels,
            extra_blocks=extra_blocks,
        )
    m = nn.Sequential(
        backbone,
        nn.Conv2d(backbone[-1].out_channels, out_channels, 1),
    )
    m.out_channels = out_channels
    return m

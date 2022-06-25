from torch import nn
from torchvision.models.detection.backbone_utils import _mobilenet_extractor

from trgnet.backbones.mobilenetv2 import mobilenet_v2
from trgnet.misc import FrozenBatchNorm2d
from trgnet.trg import TRGNet


def trgnet_mobilenet_v2(
    num_classes=91,
    **kwargs,
):
    backbone = MobileNetV2().features
    backbone = nn.Sequential(
        backbone,
        nn.Conv2d(backbone[-1].out_channels, 256, 1),
    )
    backbone.out_channels = 256

    model = TRGNet(
        backbone,
        num_classes,
        **kwargs,
    )
    return model


def trgnet_v2_mobilenet_v2(
    num_classes=91,
    pretrained_backbone=True,
    trainable_backbone_layers=3,
    **kwargs,
):
    if not pretrained_backbone:
        trainable_backbone_layers = 6

    backbone = mobilenet_v2(
        pretrained=pretrained_backbone, norm_layer=FrozenBatchNorm2d
    )
    backbone = _mobilenet_extractor(backbone, False, trainable_backbone_layers)

    anchor_sizes = (
        (
            32,
            64,
            128,
            256,
            512,
        ),
    ) * 3
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    defaults = {
        "min_size": 320,
        "max_size": 640,
    }
    kwargs = {**defaults, **kwargs}
    model = TRGNet(
        backbone,
        num_classes,
        rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios),
        **kwargs,
    )
    return model

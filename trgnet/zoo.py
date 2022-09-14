from torch import nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _mobilenet_extractor

from trgnet.backbones.mobilenetv3 import mobilenet_v3_large
from trgnet.misc import FrozenBatchNorm2d
from trgnet.training.utils import load_training
from trgnet.trg import TRGNet, TRGNetPredictor


def trgnet_mobilenet_v3_large(pretrained=False, num_classes=10, **kwargs):
    defaults = {
        "min_size": 320,
        "max_size": 640,
        "rpn_pre_nms_top_n_test": 150,
        "rpn_post_nms_top_n_test": 150,
        "rpn_score_thresh": 0.05,
    }
    kwargs = {**defaults, **kwargs}

    backbone = mobilenet_v3_large(norm_layer=FrozenBatchNorm2d)
    backbone = _mobilenet_extractor(backbone, True, 3)

    anchor_sizes = ((32, 64, 128, 256, 512),) * 3
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    model = TRGNet(
        backbone,
        num_classes,
        rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios),
        **kwargs,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = TRGNetPredictor(in_features, num_classes)

    if pretrained:
        model = load_training(model)

    return model

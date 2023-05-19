from collections import OrderedDict
from copy import copy

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign

from trgnet.anchor import AnchorGenerator
from trgnet.grpm import GaussianRegionProposal
from trgnet.rpn import RegionProposalNetwork, RPNHead
from trgnet.timer import Timer


class GeneralizedTRG(nn.Module):
    def __init__(self, backbone, rpn, roi_heads, transform, grpm):
        super().__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.grpm = grpm
        self.timer = Timer()

    def eager_outputs(self, losses, detections):
        return losses if self.training else detections

    def forward(self, frame, targets=None, use_grpm=False):
        self.timer.start("Total")
        original_image_sizes = []

        if isinstance(frame, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            images = transforms.ToTensor()(pil_img).unsqueeze_(0)
        else:
            images = frame

        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        self.timer.start("Backbone")
        features = self.backbone(images.tensors)
        self.timer.stop("Backbone")

        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        self.timer.start("RPM")
        if use_grpm:
            if not isinstance(frame, np.ndarray):
                frame = transforms.ToPILImage()(images.squeeze(0))
                frame = np.flip(np.asarray(grpm_image), 2)

            new_size = (images.tensors.shape[3], images.tensors.shape[2])
            proposals = self.grpm(frame, new_size)
            proposal_losses = {}
        else:
            proposals, proposal_losses = self.rpn(images, features, targets)
        self.timer.stop("RPM")

        self.timer.start("RoI")
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )
        self.timer.stop("RoI")

        detections = self.transform.postprocess(
            detections, images.image_sizes, original_image_sizes
        )

        losses = {}
        if self.training:
            losses.update(detector_losses)
            losses.update(proposal_losses)

        self.timer.stop("Total")
        return self.eager_outputs(losses, detections)


class TwoMLPHead(nn.Module):
    def __init__(self, in_channels, representation_size):
        super().__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


class TRGNetPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


class TRGNet(GeneralizedTRG):
    def __init__(
        self,
        backbone,
        num_classes=None,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # RPN parameters
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        # GRPM
        grpm_min_area=20,
        grpm_lr=-1,
        grpm_show_output=False,
    ):
        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(
            training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test
        )
        rpn_post_nms_top_n = dict(
            training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test
        )

        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
            )

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution**2, representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = TRGNetPredictor(representation_size, num_classes)

        roi_heads = RoIHeads(
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        grpm = GaussianRegionProposal(grpm_min_area, grpm_lr, grpm_show_output)

        super().__init__(backbone, rpn, roi_heads, transform, grpm)

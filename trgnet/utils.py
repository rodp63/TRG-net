import torch
import torchvision.transforms as transforms

from torchvision.utils import draw_bounding_boxes
from trgnet.data import Kitti


def draw_image_bb(sample_image, sample_target=None, tresh=None, dataset="kitti"):
    if dataset == "kitti":
        dataset_classes = Kitti.classes
    else:
        return

    _image = (255 * sample_image).to(torch.uint8)

    if sample_target is not None:
        _boxes = torch.stack(
            [
                sample_target["boxes"][idx]
                for idx in range(len(sample_target["boxes"]))
                if (
                    sample_target["labels"][idx] != dataset_classes.index("DontCare")
                    and (tresh is None or sample_target["scores"][idx] >= tresh)
                )
            ]
        )
        _labels = [
            sample_target["labels"][idx]
            for idx in range(len(sample_target["labels"]))
            if (
                sample_target["labels"][idx] != dataset_classes.index("DontCare")
                and (tresh is None or sample_target["scores"][idx] >= tresh)
            )
        ]
        _labels = [dataset_classes[idx] for idx in _labels]

        _image = draw_bounding_boxes(
            image=_image, boxes=_boxes, labels=_labels, width=3
        )

    _image = transforms.ToPILImage()(_image)
    _image.show()

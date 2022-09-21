import csv
import os

import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset


class Kitti(VisionDataset):
    image_dir_name = "image_2"
    labels_dir_name = "label_2"
    classes = [
        "Background",
        "Car",
        "Van",
        "Truck",
        "Pedestrian",
        "Person_sitting",
        "Cyclist",
        "Tram",
        "Misc",
        "DontCare",
    ]

    def __init__(
        self, root, train=True, transform=None, target_transform=None, _transforms=None
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            transforms=_transforms,
        )
        self.images = []
        self.targets = []
        self.root = root
        self.train = train
        self._location = "training" if self.train else "testing"

        image_dir = os.path.join(self._raw_folder, self._location, self.image_dir_name)
        if self.train:
            labels_dir = os.path.join(
                self._raw_folder, self._location, self.labels_dir_name
            )
        for img_file in os.listdir(image_dir):
            self.images.append(os.path.join(image_dir, img_file))
            if self.train:
                self.targets.append(
                    os.path.join(labels_dir, f"{img_file.split('.')[0]}.txt")
                )

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        target = self._parse_target(index) if self.train else None
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target

    def _parse_target(self, index):
        target = {}
        labels, boxes = [], []
        with open(self.targets[index]) as inp:
            content = csv.reader(inp, delimiter=" ")
            for line in content:
                labels.append(self.classes.index(line[0]))
                boxes.append([float(x) for x in line[4:8]])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target["labels"] = labels
        target["boxes"] = boxes
        target["image_id"] = torch.tensor([index])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(boxes)), dtype=torch.int64)

        return target

    def __len__(self):
        return len(self.images)

    @property
    def _raw_folder(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        root_path = os.path.join(file_path, self.root)
        return os.path.join(root_path, self.__class__.__name__, "raw")


def get_kitti_loaders(batch_size=64):
    image_transform = transforms.Compose(
        [
            transforms.Resize((375, 1242)),
            transforms.ToTensor(),
        ]
    )
    dataset = Kitti(
        root="../data",
        train=True,
        transform=image_transform,
    )
    train_set, validation_set, test_set = torch.utils.data.random_split(
        dataset, [6000, 700, 781]
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    return train_loader, validation_loader, test_loader

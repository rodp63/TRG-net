import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import draw_bounding_boxes


torch.manual_seed(17)
batch_size = 64

all_transforms = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.Kitti(
    root="./data", train=True, transform=all_transforms, download=True
)
test_dataset = torchvision.datasets.Kitti(
    root="./data", train=False, transform=all_transforms, download=True
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=True
)

print(len(train_dataset), len(test_dataset))

sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
img, labels = train_dataset[sample_idx]

print(type(img), img.shape)

boxes = [label["bbox"] for label in labels]
boxes = torch.tensor(boxes)
img = 255 * img
img = draw_bounding_boxes(
    img.to(torch.uint8), boxes, width=5, colors="green", fill=False
)

img = torchvision.transforms.ToPILImage()(img)
img.show()

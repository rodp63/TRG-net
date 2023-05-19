import torch

from trgnet.data import get_kitti_loaders
from trgnet.training.train import train
from trgnet.zoo import trgnet_mobilenet_v3_large

model = trgnet_mobilenet_v3_large(pretrained=True)
train_loader, validation_loader, test_loader = get_kitti_loaders()

train(model, train_loader, validation_loader, epochs=1, load_epoch=-1)

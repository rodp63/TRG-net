from trgnet.data import get_kitti_loaders
from trgnet.utils import draw_image_bb

from trgnet.training.train import train
from trgnet.zoo import trgnet_mobilenet_v3_large


tr, va, te = get_kitti_loaders()
images, targets = next(iter(tr))
idx = 15
draw_image_bb(images[idx], targets[idx], dataset="kitti")

model = trgnet_mobilenet_v3_large(pretrained=True)

model.eval()
pred = model([images[idx]])
draw_image_bb(images[idx], pred[0], tresh=0.7, dataset="kitti")

train(model, tr, va)

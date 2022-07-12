import cv2
import time
import torch
import imutils
import numpy as np
from imutils.video import VideoStream
from PIL import Image

from copy import copy

from trgnet.data import get_kitti_loaders
from trgnet.utils import draw_image_bb

from trgnet.training.train import train
from trgnet.zoo import trgnet_mobilenet_v3_large

import torchvision.transforms as transforms


# tr, va, te = get_kitti_loaders()
# images, targets = next(iter(va))
# idx = 15
# draw_image_bb(images[idx], targets[idx], dataset="kitti")

model = trgnet_mobilenet_v3_large(pretrained=True)
model.eval()

mean, rounds = 0, 0

with torch.no_grad():
    # print(images[idx])
    # pred = model([images[idx]])
    # draw_image_bb(images[idx], pred[0], tresh=0.0, dataset="kitti")

    # train(model, tr, va)

    video = True

    if video:
        cap = cv2.VideoCapture("data/vtest.mp4")
    else:
        cap = VideoStream(src=0).start()

    while True:
        frame = cap.read()
        if frame is None:
            break

        if video:
            frame = frame[1]

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_to_tensor = transforms.ToTensor()(pil_img).unsqueeze_(0)

        rounds += 1
        ss = time.time()
        pred = model(pil_to_tensor, True, frame)
        ee = time.time()
        mean += (ee - ss)
        print(rounds, mean)

        for idx, b in enumerate(pred[0]["boxes"]):
            if pred[0]["scores"][idx] < 0.5:
                continue
            x1, y1, x2, y2 = b[0].item(), b[1].item(), b[2].item(), b[3].item()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow("frame", frame)

    print(mean / rounds)

# cap.release()
# cv2.destroyAllWindows()

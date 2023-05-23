import sys
import time

import cv2
import torch
import torchvision
from imutils.video import VideoStream

from trgnet.data import Kitti
from trgnet.zoo import trgnet_mobilenet_v3_large

video = True
use_grpm = True
grpm_show_output = True

# Important! play with the grpm parameters
model = trgnet_mobilenet_v3_large(
    pretrained=True,
    grpm_min_area=35,
    grpm_lr=0.01,
    grpm_show_output=grpm_show_output,
)
model.eval()


if video:
    cap = cv2.VideoCapture("data/test.mp4")
else:
    cap = VideoStream(src=0).start()


with torch.no_grad():
    while True:
        frame = cap.read()
        if frame is None:
            break

        if video:
            frame = frame[1]
            if frame is None:
                break

        frame = cv2.resize(frame, (480, 320), interpolation=cv2.INTER_AREA)
        a = time.time()
        pred = model(frame, use_grpm=use_grpm)
        b = time.time()

        for idx, b in enumerate(pred[0]["boxes"]):
            if pred[0]["scores"][idx] < 0.5:
                continue

            x1, y1, x2, y2 = b[0].item(), b[1].item(), b[2].item(), b[3].item()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "{} ({:.2f})".format(
                    Kitti.classes[pred[0]["labels"][idx]], pred[0]["scores"][idx]
                ),
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow("output", frame)

        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break

    # model.timer.report()
    cap.release()
    cv2.destroyAllWindows()

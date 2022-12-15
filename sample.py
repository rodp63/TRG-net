import time

import cv2
import torch
import torchvision
import torchvision.transforms as transforms

from imutils.video import VideoStream
from PIL import Image

from trgnet.zoo import trgnet_mobilenet_v3_large
from trgnet.data import Kitti

video = True
use_grpm = True
grpm_show_output = True

model = trgnet_mobilenet_v3_large(
    pretrained=True, grpm_min_area=50, grpm_lr=0.01, grpm_show_output=grpm_show_output
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

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_to_tensor = transforms.ToTensor()(pil_img).unsqueeze_(0)

        a = time.time()
        pred = model(pil_to_tensor, use_grpm=use_grpm, grpm_image=frame)
        b = time.time()

        for idx, b in enumerate(pred[0]["boxes"]):
            if pred[0]["scores"][idx] < 0.4:
                continue
            print(Kitti.classes[pred[0]["labels"][idx]], pred[0]["scores"][idx])
            x1, y1, x2, y2 = b[0].item(), b[1].item(), b[2].item(), b[3].item()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow("frame", frame)

        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break

    # model.timer.report()
    cap.release()
    cv2.destroyAllWindows()

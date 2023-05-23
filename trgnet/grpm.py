import cv2
import imutils
import numpy as np
import torch
import torch.nn as nn


class GaussianRegionProposal(nn.Module):
    def __init__(self, min_area, lr, show_output):
        super().__init__()
        self.min_area = min_area
        self.lr = lr
        self.show = show_output
        self.gmm = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def forward(self, image, new_size, save_output=False):
        boxes = []
        if new_size != (image.shape[1], image.shape[0]):
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

        mask = self.gmm.apply(image, learningRate=self.lr)
        mask[mask < 255] = 0

        img_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        contours, hierarchy = cv2.findContours(
            img_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        for cnt in contours:
            if cv2.contourArea(cnt) >= self.min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append(torch.Tensor([x, y, x + w, y + h]))
                for t in range(1, 3):
                    boxes.append(torch.Tensor([x - t, y - t, x + w + t, y + h + t]))
                    boxes.append(
                        torch.Tensor(
                            [x - t - 2, y - t - 2, x + w + t + 2, y + h + t + 2]
                        )
                    )
                if self.show:
                    cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if len(boxes) == 0:
            boxes = [torch.Tensor([0, 0, 1, 1])]

        if self.show:
            cv2.imshow("mask", mask)
            if save_output:
                cv2.imwrite("mask.png", mask)

        return [torch.stack(boxes)]

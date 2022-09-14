import cv2
import imutils
import numpy as np
import torch
import torch.nn as nn


class GaussianRegionProposal(nn.Module):
    def __init__(self, min_area=20, lr=0.01):
        super().__init__()
        self.min_area = min_area
        self.lr = lr
        self.gmm = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def forward(self, image):
        boxes = []
        mask = self.gmm.apply(image, learningRate=self.lr)
        mask[mask < 255] = 0

        img_close = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        contours, hierarchy = cv2.findContours(
            img_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            if cv2.contourArea(cnt) >= self.min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append(torch.Tensor([x, y, x + w, y + h]))

        if len(boxes) == 0:
            boxes = [torch.Tensor([0, 0, 1, 1])]

        return [torch.stack(boxes)]

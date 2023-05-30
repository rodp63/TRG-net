import math

import cv2
import imutils
import torch
import torch.nn as nn


class GaussianRegionProposal(nn.Module):
    def __init__(self, min_area, lr, show_output):
        super().__init__()
        self.lr = lr
        self.show = show_output
        self.gmm = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def pick_k_best(self, _list, k):
        return sorted(_list, key=lambda it: it[2] * it[3])[-k:]

    def get_contours(self, mask_list):
        contour_list = []
        for mask in mask_list:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contour_list.append(contours)
        return contour_list

    def gather_boxes(self, boxes, gather_idx, max_area_ratio=1.5):
        new_boxes = set()
        if boxes:
            tot = len(boxes) // 2
            max_area = sum([b[2] * b[3] for b in boxes[-tot:]]) / tot * max_area_ratio
        for box1 in boxes:
            candidates = []
            for box2 in boxes:
                if box1 == box2:
                    continue
                x1, y1 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
                x2, y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
                distance = math.sqrt((x1 - x2) * (x1 - x2) + (y2 - y1) * (y2 - y1))
                candidates.append((distance, box2))
            candidates = sorted(candidates, key=lambda it: it[0])[:gather_idx]
            for sc, box2 in candidates:
                x1, y1, w1, h1 = box1
                x2, y2, w2, h2 = box2
                new_x, new_y = min(x1, x2), min(y1, y2)
                new_w = max(x1 + w1, x2 + w2) - new_x
                new_h = max(y1 + h1, y2 + h2) - new_y
                if new_w * new_h < max_area:
                    new_box = (new_x, new_y, new_w, new_h)
                    new_boxes.add(new_box)
        return list(new_boxes)

    def get_boxes(self, contours, max_boxes=70, gather=True, gather_idx=3):
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))
        boxes = self.pick_k_best(boxes, max_boxes)
        if gather:
            boxes += self.gather_boxes(boxes, gather_idx)
        return boxes

    def forward(self, image, new_size, save_output=False, max_boxes=35, expand=2):
        boxes, _boxes = [], []
        if new_size != (image.shape[1], image.shape[0]):
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

        mask = self.gmm.apply(image, learningRate=self.lr)
        mask[mask < 255] = 0

        open_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        close_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=3)
        if self.show:
            cv2.imshow("open mask", open_mask)
            cv2.imshow("close mask", close_mask)

        contours, open_contours, close_contours = self.get_contours(
            [mask, open_mask, close_mask]
        )
        _boxes = self.get_boxes(contours)
        _open_boxes = self.get_boxes(open_contours)
        _close_boxes = self.get_boxes(close_contours)

        _all_boxes = _boxes + _open_boxes + _close_boxes
        _all_boxes = self.pick_k_best(_all_boxes, max_boxes * 2)
        _all_boxes += self.gather_boxes(_all_boxes, 3)
        _all_boxes = self.pick_k_best(_all_boxes, max_boxes)
        for box in _all_boxes:
            x, y, w, h = box
            for t in range(0, expand):
                t *= 2
                boxes.append(torch.Tensor([x - t, y - t, x + w + 2 * t, y + h + 2 * t]))

        if len(boxes) == 0:
            boxes = [torch.Tensor([0, 0, 1, 1])]

        if self.show:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            for box in boxes:
                x, y, x1, y1 = box
                x, y, x1, y1 = int(x), int(y), int(x1), int(y1)
                cv2.rectangle(mask, (x, y), (x1, y1), (0, 255, 0), 1)
                # cv2.drawContours(mask, [cnt], 0, (0,0,255), 2)
            cv2.imshow("mask", mask)
            if save_output:
                cv2.imwrite("mask.png", mask)

        return [torch.stack(boxes)]

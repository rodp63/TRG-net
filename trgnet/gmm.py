import cv2
import imutils
import numpy as np
from imutils.video import VideoStream

min_length = 150
min_area = 20
learning_rate = 0.05
video = True

if video:
    cap = cv2.VideoCapture("data/vtest.mp4")
else:
    cap = VideoStream(src=0).start()

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

while True:
    frame = cap.read()
    if video:
        frame = frame[1]

    frame = imutils.resize(frame, width=500)

    if frame is None:
        break

    fgmask = fgbg.apply(frame, learningRate=learning_rate)

    img_close = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(
        img_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        # length = cv2.arcLength(cnt, True)
        # if length > min_length:
        if cv2.contourArea(cnt) >= min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("frame", frame)
    cv2.imshow("mask", fgmask)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

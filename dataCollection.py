import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

capture = cv2.VideoCapture(0)  # for capturing image
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = "images/Z"
counter = 0

while True:
    success, img = capture.read()
    hands, img = detector.findHands(img)
    #  for crop the image
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']  # bounding box

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        # starting value to ending the image to put it in the white

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))  # for actual size of img
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)  # to white image appear in the middle
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / h
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))  # for actual size of img
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)  # to white image appear in the middle
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("Image-Crop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image from Webcam", img)

    key = cv2.waitKey(1)  # wait will be 1 millisecond
    if key == ord("h"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)

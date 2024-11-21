import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

capture = cv2.VideoCapture(0)  # for capturing image
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 3001

folder = "images"
counter = 0

labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

while True:
    success, img = capture.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img, draw=False)
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
            prediction, index = classifier.getPrediction(imgWhite, draw=False)  # will give prediction and index
            print(prediction, index)

        else:
            k = imgSize / h
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))  # for actual size of img
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)  # to white image appear in the middle
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)  # to stop draw in the video

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),
                      (255, 255, 255), cv2.FILLED)

        cv2.putText(imgOutput, labels[index], (x, y - 27), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 0, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow("Image-Crop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image from Webcam", imgOutput)

    cv2.waitKey(1)  # wait will be 1 millisecond
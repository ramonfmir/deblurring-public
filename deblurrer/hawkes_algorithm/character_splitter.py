import sys

import numpy as np
import cv2

from blurrer import perspective_transformation as pt

img = cv2.imread(sys.argv[1])

# Image pre-processing.

img = pt.apply_perspective(img, 10, 0)

# Finding characters.

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

rectangles = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 30 and area < 700:
        rectangles.append(cv2.boundingRect(cnt))

rectangles, _ = cv2.groupRectangles(rectangles, 0, 0.5)

for rect in rectangles:
    [x, y, w, h] = rect
    if  h > 16 and h > w:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        roi = thresh[y:(y + h), x:(x + w)]
        roismall = cv2.resize(roi, (10, 10))

cv2.imshow('Characters', img)
cv2.waitKey(0)

# TODO: Splitting into regions.

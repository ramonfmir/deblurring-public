import sys

import numpy as np
import cv2

from blurrer import perspective_transformation as pt

# Percentage tolerance of intersection
TOLERANCE = 0.15

def contain(rectA, rectB):
    x1,y1,w1,h1 = rectA
    x2,y2,w2,h2 = rectB

    return (x1 < x2) and (y1 < y2) and (x1 + w1 > x2 + w2) and (y1 + h1 > y2 + h2)

# Helper methods
def intersection(rectA, rectB):
    # Pre -- Assume non-zero rectangles
    x1,y1,w1,h1 = rectA
    x2,y2,w2,h2 = rectB

    intersectWidth  = 0
    intersectHeight = 0
    # Check if widths intersect
    if x1 <= x2:
        if (x2 - x1) >= w1:
            return False
        intersectWidth = x1 + w1 - x2
    else:
        if (x1 - x2) >= w2:
            return False
        intersectWidth = x2 + w2 - x1

    # Check if heights intersect
    if y1 <= y2:
        if (y2 - y1) >= h1:
            return False
        intersectHeight = y1 + h1 - y2
    else:
        if (y1 - y2) >= h2:
            return False
        intersectHeight = y2 + h2 - y1

    if contain(rectA, rectB) or contain(rectB, rectA):
        return True

    intersect_area = float(intersectWidth * intersectHeight)
    area_of_both = rect_area(rectA) + rect_area(rectB) - intersect_area

    # Check tolerance
    if (intersect_area / area_of_both) < TOLERANCE:
        return False

    return True

def rect_area(rect):
    x,y,w,h = rect
    return w * h

def group_rectangles(rectangles):
    grouped = []
    for rectA in rectangles:
        intersected = False
        for rectB in grouped:
            if intersection(rectA, rectB):
                print(rectA,rectB)
                intersected = True
                areaA = rect_area(rectA)
                areaB = rect_area(rectB)
                if areaA > areaB:
                    grouped.remove(rectB)
                    grouped.append(rectA)
                    break

        if not intersected:
            grouped.append(rectA)

    # Return list of non-intersecting rectangles
    return grouped

if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])

    # Image pre-processing.

    img = pt.apply_perspective(img, 13, 0)

    # Finding characters.

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    all_rectangles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        rect = cv2.boundingRect(cnt)
        if area > 40 and area < 700:
            all_rectangles.append(rect)

    rectangles = group_rectangles(all_rectangles)

    for rect in rectangles:
        [x, y, w, h] = rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        roi = thresh[y:(y + h), x:(x + w)]
        roismall = cv2.resize(roi, (10, 10))

    cv2.imshow('Characters', img)
    cv2.waitKey(0)

# TODO: Splitting into regions.

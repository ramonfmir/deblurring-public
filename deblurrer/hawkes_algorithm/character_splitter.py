import sys

import numpy as np
import cv2

from blurrer import perspective_transformation as pt

# Helper methods
def group_rectangles(rectangles):
    return _group_rectangles(rectangles, [])

def _group_rectangles(input_rectangles, output_rectangles):
    if not input_rectangles:
        return output_rectangles

    curr_rect = input_rectangles.pop()
    has_intersected = False
    for i in range(len(output_rectangles)):
        other_rect = output_rectangles[i]
        if intersection(curr_rect, other_rect):
            has_intersected = True
            area_curr = cv2.contourArea(curr_rect)
            area_other = cv2.contourArea(curr_rect)
            if area_curr > area_other:
                del other_rect
                output_rectangles.append(curr_rect)
                has_intersected = False
                break

    if not has_intersected:
        output_rectangles.append(curr_rect)

    return _group_rectangles(input_rectangles, output_rectangles)

if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])

    # Image pre-processing.

    img = pt.apply_perspective(img, 10, 0)

    # Finding characters.

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    all_rectangles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 30 and area < 700:
            all_rectangles.append(cv2.boundingRect(cnt))

    rectangles = group_rectangles(all_rectangles)
    for rect in rectangles:
        [x, y, w, h] = rect
        if  h > 16 and h > w:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            roi = thresh[y:(y + h), x:(x + w)]
            roismall = cv2.resize(roi, (10, 10))

    cv2.imshow('Characters', img)
    cv2.waitKey(0)

# TODO: Splitting into regions.

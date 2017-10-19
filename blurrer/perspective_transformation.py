# Experiment with perspective transformation. Potentially useful for data
# generation. Usage:
#
# python perspective_transformation.py image margin_right margin_left

import sys

import numpy as np
import cv2

img_name = sys.argv[1]
margin_right = int(sys.argv[2])
margin_left  = int(sys.argv[3])

img = cv2.imread(img_name)

h, w, _ = img.shape

pts_i = np.float32([[0, 0],
                    [0, h],
                    [w, 0],
                    [w, h]])
pts_o = np.float32([[0, margin_left],
                    [0, h - margin_left],
                    [w, margin_right],
                    [w, h - margin_right]])

M = cv2.getPerspectiveTransform(pts_i, pts_o)

img = cv2.warpPerspective(img, M, (w, h))

cv2.imshow('Perspective', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

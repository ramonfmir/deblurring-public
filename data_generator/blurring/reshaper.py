import sys
import math
import numpy as np
import cv2
import scipy.misc as sc
import random as rand
from numpy.random import randint

# Applies a 'perspective' transformation. The pov is a number between -1 and 1.
# It will affect the angle of the perspective.
def apply_perspective(pov, img):
    height, width, _ = img.shape

    angle = pov * math.atan(height / width) / 2
    top_left, bottom_left, top_right, bottom_right = (0,) * 4

    if angle > 0:
        top_left, bottom_right = (width * math.tan(angle),) * 2
    else:
        top_right, bottom_left = (width * math.tan(-angle),) * 2

    pts_i = np.float32([[0, 0],
                        [0, height],
                        [width, 0],
                        [width, height]])
    pts_o = np.float32([[0, top_left],
                        [0, height - bottom_left],
                        [width, top_right],
                        [width, height - bottom_right]])

    val = int(rand.beta(0.5, 0.5) * 255)
    M = cv2.getPerspectiveTransform(pts_i, pts_o)
    img = cv2.warpPerspective(img, M, (width, height),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=[0, val, 0])

    return img


# Reduce the size of the image. The magnitude is between 0 and 1. 0 is
# equivalent to reducing the size by a half and 1 is equivalent to keeping the
# image as it is.
def reduce_size(magnitude, img):
    scale = 0.5 + (magnitude / 2)
    height, width, _ = img.shape
    new_height, new_width = int(scale * height), int(scale * width)
    v_border = (height - new_height) / 2
    h_border = (width - new_width) / 2

    val = int(rand.beta(0.5, 0.5) * 255)
    img = cv2.resize(img, (new_width, new_height))
    new_img = cv2.copyMakeBorder(img, math.ceil(v_border), math.floor(v_border),
                                      math.ceil(h_border), math.floor(h_border),
                                 cv2.BORDER_CONSTANT,value=[0, val, 0])

    return new_img


# Apply rotation.
def rotate_image(angle, img):
    return sc.imrotate(img, angle)


if __name__ == "__main__":
    img = cv2.imread("../tests/license_plate3.jpg")
    # random_border(img)

    img = reduce_size(0.95, apply_perspective(0.05, img))
    cv2.imshow('Perspective', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

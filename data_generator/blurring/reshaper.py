import sys
import math
import numpy as np
import cv2
import scipy.misc as sc
import random as rand
from numpy.random import randint
from PIL import Image
from copy import deepcopy

# Applies a 'perspective' transformation. The pov is a number between -1 and 1.
# It will affect the angle of the perspective.
def apply_perspective(pov, val, img):
    height, width, _ = img.shape

    angle = pov * math.atan(height / width) / 2
    top_left, bottom_left, top_right, bottom_right = (0,) * 4

    if angle > 0:
        top_left, bottom_right = (width * math.tan(angle),) * 2
    else:
        top_right, bottom_left = (width * math.tan(-angle),) * 2

    margin_left = (top_left + bottom_left)
    margin_right = (top_right + bottom_right) 

    pts_i = np.float32([[0, 0],
                        [0, height],
                        [width, 0],
                        [width, height]])
    pts_o = np.float32([[margin_left, top_left],
                        [margin_left, height - bottom_left],
                        [width - margin_right, top_right],
                        [width - margin_right, height - bottom_right]])

    border_val = border_colour()
    M = cv2.getPerspectiveTransform(pts_i, pts_o)
    img = cv2.warpPerspective(img, M, (width, height),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=[val, val, val])

    return img


# Reduce the size of the image. The magnitude is between 0 and 1. 0 is
# equivalent to reducing the size by a half and 1 is equivalent to keeping the
# image as it is.
def reduce_size(magnitude, val, img):
    scale = 0.5 + (magnitude / 2)
    height, width, _ = img.shape
    new_height, new_width = int(scale * height), int(scale * width)
    v_border = (height - new_height) / 2
    h_border = (width - new_width) / 2
    
    img = cv2.resize(img, (new_width, new_height))

    border_val = border_colour()

    new_img = cv2.copyMakeBorder(img, math.ceil(v_border), math.floor(v_border),
                                      math.ceil(h_border), math.floor(h_border),
                                 cv2.BORDER_CONSTANT,value=[val, val, val])

    return new_img


def pad_with_val_and_scale(max_size, val, img):
    img_height, img_width, _ = img.shape
    scale = min(max_size[1] / img_height, max_size[0] / img_width)
    new_width, new_height = int(scale * img_width), int(scale * img_height)
    img = cv2.resize(img, (new_width, new_height))

    h_border = (max_size[0] - new_width) / 2
    v_border = (max_size[1] - new_height) / 2

    new_img = cv2.copyMakeBorder(img, math.ceil(v_border), math.floor(v_border),
                                      math.ceil(h_border), math.floor(h_border),
                                 cv2.BORDER_CONSTANT,value=[val, val, val])

    return new_img

# Make vertical and horizontal colour different
# Make it a bit bigger than original

# Apply rotation.
def reshape(magnitude, pov, width, height, img):
    val = int(np.random.beta(0.5, 0.5) * 255)
    max_size = [width, height]
    img = pad_with_val_and_scale(max_size, val, img)
    img = apply_perspective(pov, val, img)
    img = reduce_size(magnitude, val, img)
    return img, deepcopy(img)


if __name__ == "__main__":
    img = cv2.imread("../tests/lp.jpg")
    # random_border(img)

    img = reshape(0.7, 0.5, 256, 80, img)
    cv2.imshow('Perspective', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

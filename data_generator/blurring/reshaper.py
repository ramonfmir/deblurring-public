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

    pts_i = np.float32([[0, 0],
                        [0, height],
                        [width, 0],
                        [width, height]])
    pts_o = np.float32([[0, top_left],
                        [0, height - bottom_left],
                        [width, top_right],
                        [width, height - bottom_right]])

    M = cv2.getPerspectiveTransform(pts_i, pts_o)
    img = cv2.warpPerspective(img, M, (width, height),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=[0, val, 0])

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
    new_img = cv2.copyMakeBorder(img, math.ceil(v_border), math.floor(v_border),
                                      math.ceil(h_border), math.floor(h_border),
                                 cv2.BORDER_CONSTANT,value=[0, val, 0])

    return new_img


def pad_with_val(max_size, val, old_im):
    old_height, old_width, _ = old_im.shape

    new_size_w = int((max_size[0] - old_width) / 2)
    new_size_h = int((max_size[1] - old_height) / 2)

    new_img = Image.new("RGB", max_size, "rgb(0, " + str(val) + ", 0)")
    new_img.paste(Image.fromarray(old_im), (new_size_w, new_size_h))

    return new_img

def pad_with_val_and_scale(max_size, val, img):
    img_height, img_width, _ = img.shape
    scale = min(max_size[1] / img_height, max_size[0] / img_width)
    new_img = cv2.resize(img, (int(scale * img_width), int(scale * img_height)))
    new_img = pad_with_val(max_size, val, new_img)
    return np.array(new_img)

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

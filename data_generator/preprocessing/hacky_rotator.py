import os
import sys

import math
import numpy as np
import cv2
import scipy.misc as sc
from numpy.random import randint

img_type = "JPEG"

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
    new_img = cv2.copyMakeBorder(img, math.ceil(v_border), math.floor(v_border),
                                      math.ceil(h_border), math.floor(h_border),
                                 cv2.BORDER_CONSTANT,value=[val, val, val])

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


def hack_the_hack(img_dirty, img_hack):
    magnitude = float(input("Resize Factor [0, 1] \n"))

    val = 0 if str(input("Input Border Colour {B, W} \n")) == "B" else 255
    img_hack = reduce_size(magnitude, val, img_hack)
    pov = float(input("Input Perspective Transformation [-1, 1] \n"))
    img_hack = apply_perspective(pov, val, img_hack)

    angle = float(input("Input Rotation Angle [0, 180] \n"))
    img_hack = sc.imrotate(img_hack, angle)

    return img_hack


if __name__ == "__main__":
    directory_dirty = sys.argv[1]
    directory_hack  = sys.argv[2]
    directory_out   = sys.argv[3]

    for filename in os.listdir(directory_dirty):
        img_dirty_path = directory_dirty + "/" + str(filename)
        img_hack_path = directory_hack + "/" + "generated" + str(filename)
        file, ext = os.path.splitext(img_hack_path)

        img_dirty = cv2.imread(img_dirty_path)
        if(os.path.exists(img_hack_path)):
            img_hack  = cv2.imread(img_hack_path)
        else:
            continue

        cv2.imshow("Dirty Image", img_dirty)
        cv2.imshow("Dirty HACK Image", img_hack)
        cv2.waitKey(0)

        if input("Skip [Y/N]") == "Y":
            continue

        redo = True
        while(redo) :
            new_img = hack_the_hack(img_dirty, img_hack)
            cv2.destroyAllWindows()
            cv2.imshow("Dirty Image", img_dirty)
            cv2.imshow("New Image", new_img)
            cv2.waitKey(0)
            redo = True if input("Redo [Y/N] \n") == "Y" else False

        cv2.destroyAllWindows()
        new_img_path = directory_out + "/" + str(filename)
        cv2.imwrite(new_img_path, new_img)

    cv2.destroyAllWindows()

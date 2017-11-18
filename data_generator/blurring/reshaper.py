import sys
import math
import numpy as np
import cv2
import scipy.misc as sc
from numpy.random import randint

# Applies a 'perspective' transformation. The pov is a number between -1 and 1.
# It will affect the angle of the perspective.
def apply_perspective(pov, img, seed):
    np.random.seed(seed)
    height, width, _ = img.shape

    fill = border_colour()
    #horizontal
    horiz = randint(2,6)
    np.place(img[:,0:horiz], img[:,0:horiz] < 20, fill)
    np.place(img[:,-horiz:], img[:,-horiz:] < 20, fill)

    #vertical
    vert = randint(2,13)
    np.place(img[0:vert,:], img[0:vert,:] < 20, fill)
    np.place(img[-vert:,:], img[-vert:,:] < 20, fill)

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

    border_val = border_colour()
    M = cv2.getPerspectiveTransform(pts_i, pts_o)
    img = cv2.warpPerspective(img, M, (width, height),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=border_val)

    return img


# Reduce the size of the image. The magnitude is between 0 and 1. 0 is
# equivalent to reducing the size by a half and 1 is equivalent to keeping the
# image as it is.
def reduce_size(magnitude, img, seed):
    np.random.seed(seed)
    scale = 0.5 + (magnitude / 2)
    height, width, _ = img.shape
    new_height, new_width = int(scale * height), int(scale * width)
    v_border = (height - new_height) / 2
    h_border = (width - new_width) / 2

    img = cv2.resize(img, (new_width, new_height))

    border_val = border_colour()

    new_img = cv2.copyMakeBorder(img, math.ceil(v_border), math.floor(v_border),
                                      math.ceil(h_border), math.floor(h_border),
                                       cv2.BORDER_CONSTANT, value=border_val)

    return new_img

def border_colour():
    p = np.random.random()
    if p < 0.33:
        #dark
        return randint(0,50,(3)).tolist()
    elif p < 0.66:
        #medium
        return randint(100,140,(3)).tolist()
    else:
        #bright
        return randint(220,255,(3)).tolist()

# Fill black borders (created by reduce_size, apply_perspective) with random noise
def random_border(img):
    bright_green = np.array([0, 255, 0])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = img[i, j]
            if np.array_equal(pixel, bright_green):
                img[i, j] = randint(0,255,(3)).tolist()

    return img

# Make vertical and horizontal colour different
# Make it a bit bigger than original

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

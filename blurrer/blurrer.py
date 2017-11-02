# inspired by:
# * http://www.geeksforgeeks.org/opencv-python-program-to-blur-an-image/
# * https://www.packtpub.com/mapt/book/application_development/9781785283932/2/ch02lvl1sec21/motion-blur
import sys

import cv2
import numpy as np
import random as rand
from functools import partial

# Averaging Blur. This is done by replacing each pixel by the mean of its
# kernel neighbours. We take the kernel here to be (acer)x(aver)
def averaging_blur(k_size, img):
    img = cv2.blur(img, (k_size, k_size))
    return img

# Gaussian Blur. This takes the kernel size and the standard deviation of the
# Gaussian distribution.
def gaussian_blur(k_size, s_div, img):
    img = cv2.GaussianBlur(img, (k_size, k_size), s_div)
    return img

# Median Blur. This assumes a square kernel of size k_size and replaces each
# pixel with the median value of its neighboring pixels.
def median_blur(k_size, img):
    img = cv2.medianBlur(img, k_size)
    return img

# Bilateral Filtering. This takes in two parameters, the sigma colour and the
# sigma size (essentially kernel size). The sigma colour parameter determines
# how much colours are mixed in each kernel.
def bilateral_blur(k_size, s_colour, img):
    # Default a larger kernel size value because of how the function works.
    # In the doc it is mentioned that k_size < 10 (which would be a typical
    # input) the function has no effect.
    k_size = 75 if k_size < 75 else  k_size;
    img = cv2.bilateralFilter(img, s_colour, k_size, k_size)
    return img

# Motion Blur.
def motion_blur(k_size, motion_direction, img):
    kernel_motion_blur = np.zeros((k_size, k_size))
    if (motion_direction == 'H'):
        kernel_motion_blur[:, int((k_size - 1) / 2)] = np.ones(k_size)
        kernel_motion_blur = kernel_motion_blur / k_size
    img = cv2.filter2D(img, -1, kernel_motion_blur)
    return img

def pixellate_blur(im, magnitude):
    height, width = im.shape[:2]
    half_img =  cv2.resize(im, (int(width/magnitude), int(height/magnitude)))
    quality_blur = cv2.resize(half_img, (width, height),interpolation = cv2.INTER_NEAREST)
    return quality_blur

def apply_blurs_randomly(blurs, img):
    rand_blurs = list(blurs)
    rand.shuffle(rand_blurs)
    for blur in rand_blurs:
        img = blur(img)
    return img

if __name__ == "__main__":
    img_name   = sys.argv[1]
    # Kernel size for averaging blur.
    averaging_k = int(sys.argv[2])
    # Kernel size for gaussian blur.
    gaussian_k  = int(sys.argv[3])
    # Standard Deviation for gaussian blur.
    gaussian_s  = int(sys.argv[4])
    # Kernel size for median blur
    median_k    = int(sys.argv[5])
    # Kernel size for bilateral blur.
    bilateral_k = int(sys.argv[6])
    # Sigma colour for bilateral blur.
    bilateral_c = int(sys.argv[7])
    # Kernel size for motion blur.
    motion_k    = int(sys.argv[8])
    # TODO
    # Direction for motion blur.
    # motion_d    = float(sys.argv[9])

    img = cv2.imread(img_name)

    averaging_b = partial(averaging_blur, averaging_k)
    median_b    = partial(median_blur, median_k)
    gaussian_b  = partial(gaussian_blur, gaussian_k, gaussian_s)
    bilateral_b = partial(bilateral_blur, bilateral_k, bilateral_c)
    motion_b    = partial(motion_blur, motion_k, 'H')

    blurs = [averaging_b, median_b, gaussian_b, bilateral_b, motion_b]

    img = apply_blurs_randomly(blurs, img)

    cv2.imshow('Blur', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

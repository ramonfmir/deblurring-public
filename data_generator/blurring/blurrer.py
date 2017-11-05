# inspired by:
# * http://www.geeksforgeeks.org/opencv-python-program-to-blur-an-image/
# * https://sourceforge.net/p/octave/image/ci/default/tree/inst/fspecial.m#l379
# * https://www.packtpub.com/mapt/book/application_development/9781785283932/2/ch02lvl1sec21/motion-blur
import sys

import cv2
import numpy as np
import scipy.misc as sc
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
    k_size = 75 if k_size < 75 else k_size;
    img = cv2.bilateralFilter(img, s_colour, k_size, k_size)
    return img

def pixelate_blur(magnitude, img):
    height, width, _  = img.shape

    new_height = int(height/magnitude)
    new_width  = int(width/magnitude)
    new_img = cv2.resize(img, (new_width, new_height))

    pixeled_img = cv2.resize(new_img, (width, height),
                             interpolation = cv2.INTER_NEAREST)
    return pixeled_img

def apply_blurs_randomly(blurs, img):
    rand_blurs = list(blurs)
    rand.shuffle(rand_blurs)
    for blur in rand_blurs:
        img = blur(img)
    return img

def motion_blur(size, theta, img):
    # First generate a horizontal line across the middle
    kernel = np.zeros([size, size])
    kernel[int(np.floor(size/2))][0:size] = 1

    # Then rotate to specified angle
    kernel = sc.imrotate(kernel,theta)
    kernel = kernel / np.sum(kernel);

    img =  cv2.filter2D(img, -1, kernel)
    return img

if __name__ == "__main__":
    img = cv2.imread('tests/license_plate.jpg')
    img = motion_blur(9, 45, img)
    cv2.imshow('name', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
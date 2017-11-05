# Adapted from:
# https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
import sys

import cv2
import numpy as np

# Simple Histogram Equalisation. Is not perfect when a specific image
# region has a much higher/lower contrast than the rest of the image.
def hist_equalise(img):
    equ = cv2.equalizeHist(img)
    return equ


# Contrast Limited Adaptive Histogram Equalisation. This breaks the 
# image into t_size x t_size squares and applies what is essentially
# equalizeHist()to each of them. 
# The contrast_lim parameter is used to avoid amplifying noise. If a histogram
# bin has contrast > contrast_lim then the pixels are clipped and distributed 
#uniformly to other bins.
def clahe(img, t_size, c_lim):
    clahe =  cv2.createCLAHE(clipLimit=c_lim, tileGridSize=(t_size,t_size))
    equ = clahe.apply(img)
    return equ

if __name__ == "__main__":
    img_name = sys.argv[1]
    img  = cv2.imread(img_name, 0)
    cv2.imshow('Original', img)

    img1 = hist_equalise(img)
    cv2.imshow('Simple', img1)

    tile_size = 3
    contrast_lim = 5
    img2 = clahe(img, tile_size, contrast_lim)
    cv2.imshow('CLAHE', img2)
    cv2.waitKey(0)


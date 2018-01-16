import cv2
import numpy as numpy

def increase_contrast(img, level=18):
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(level,level))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)  # convert from RGB to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2,a,b))  # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)  # convert from LAB to RGB
    return img2

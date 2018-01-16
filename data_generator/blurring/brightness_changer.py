import sys
import numpy as np
import cv2
import random as rand

def increase_brightness(amount, img):
    img = np.where((255 - img) < amount, 255, img + amount)
    return img

def decrease_brightness(amount, img):
    img = np.where(img < amount, 0, img - amount)
    return img

def change_brightness(amount, img):
    if rand.randint(0, 1) == 0:
        return decrease_brightness(amount, img)
    else:
        return increase_brightness(amount, img)

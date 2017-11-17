import sys
import numpy as np
import cv2

def change_brightness(amount, img):
    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #hsv[:,:,2] += amount
    #img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img = np.where((255 - img) < amount, 255, img + amount)
    return img

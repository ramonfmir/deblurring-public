import numpy as np
import cv2

# Make the char_detector's life easier.
def char_enhance(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

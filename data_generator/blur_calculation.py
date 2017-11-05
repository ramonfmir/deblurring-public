# Different metrics to detect blur in images. Adapted from: 
# https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/

import sys
import numpy as np
import cv2


def simple_detection(img):
   return cv2.Laplacian(img, cv2.CV_64F).var()

if __name__ == "__main__":
    img_name   = sys.argv[1]
    img = cv2.imread(img_name)
    print(float(simple_detection(img)))

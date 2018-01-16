import cv2
import numpy as np
import os

def normalize(folder):
    abspath = os.path.abspath(folder)
    white = [255, 255, 255]
    for f in os.listdir(abspath):
        img_path = os.path.abspath(abspath + "/" + f)
        img = cv2.imread(img_path)

        # Convert BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define range of blue color in HSV
        lower_blue = np.array([50,60,30])
        upper_blue = np.array([255,255,255])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        rand_nudge = np.random.randint(-30, 30, (3))
        blue = rand_nudge + [216, 90, 70]
        img[np.where((np.logical_and(hsv >= lower_blue, hsv <= upper_blue)).all(axis=2))] = blue

        lower_white = np.array([0,0,0])
        upper_white = np.array([360,40,260])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_white, upper_white)

        img[np.where((np.logical_and(hsv >= lower_white, hsv <= upper_white)).all(axis=2))] = white

        cv2.imwrite(img_path, img)

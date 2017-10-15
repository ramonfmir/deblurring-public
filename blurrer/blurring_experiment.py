# Inspired by:
# * http://www.geeksforgeeks.org/opencv-python-program-to-blur-an-image/
# * https://www.packtpub.com/mapt/book/application_development/9781785283932/2/ch02lvl1sec21/motion-blur

import cv2
import numpy as np
img = cv2.imread('license_plate.jpg')

# Averaging
avging = cv2.blur(img, (10, 10))
cv2.imshow('Averaging',avging)
cv2.waitKey(0)

# Gaussian Blurring
gausBlur = cv2.GaussianBlur(img, (5, 5),0)
cv2.imshow('Gaussian Blurring', gausBlur)
cv2.waitKey(0)

# Median blurring
medBlur = cv2.medianBlur(img, 5)
cv2.imshow('Media Blurring', medBlur)
cv2.waitKey(0)

# Bilateral Filtering
bilFilter = cv2.bilateralFilter(img, 9, 75, 75)
cv2.imshow('Bilateral Filtering', bilFilter)
cv2.waitKey(0)

# Motion blurring
size = 10
kernel_motion_blur = np.zeros((size, size))
kernel_motion_blur[:, int((size - 1) / 2)] = np.ones(size)
kernel_motion_blur = kernel_motion_blur / size
output = cv2.filter2D(img, -1, kernel_motion_blur)
cv2.imshow('Motion Blur', output)
cv2.waitKey(0)

cv2.destroyAllWindows()

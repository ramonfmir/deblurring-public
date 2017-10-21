# inspired by:
# * http://www.geeksforgeeks.org/opencv-python-program-to-blur-an-image/
# * https://www.packtpub.com/mapt/book/application_development/9781785283932/2/ch02lvl1sec21/motion-blur
import sys

import cv2
import numpy as np

def apply_blur(img, aver, gaussian, median, motion):
    # Averaging
    img = cv2.blur(img, (10, 10))

    # Gaussian Blurring
    img = cv2.GaussianBlur(img, (5, 5),0)

    # Median blurring
    img = cv2.medianBlur(img, 5)

    # Bilateral Filtering
    img = cv2.bilateralFilter(img, 9, 75, 75)

    # Motion blurring
    size = 10
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[:, int((size - 1) / 2)] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    img = cv2.filter2D(img, -1, kernel_motion_blur)

    return img

if __name__ == "__main__":
    img_name = sys.argv[1]
    aver     = int(sys.argv[2])
    gaussian = int(sys.argv[3])
    median   = int(sys.argv[4])
    motion   = int(sys.argv[5])

    img = cv2.imread(img_name)

    img = apply_blur(img, aver, gaussian, median, motion)

    cv2.imshow('Blur', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

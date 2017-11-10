import cv2
import numpy as numpy

def increase_contrast(img):
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2,a,b))  # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    return img2

if __name__ == "__main__":
    img = cv2.imread("license_plate.jpg")
    print(img)
    img = increase_contrast(img)

    cv2.imshow('contrast', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import data_generator.blurring.blurrer as bl
import data_generator.blurring.reshaper as rs
import data_generator.blurring.contrast as ct

import cv2
import numpy as np
import random as rand
import input_data
from copy import deepcopy

def corrupt(img):
    gaussian_kernel_size = kernel_size_corrector(rand.randint(6, 21))
    gaussian_sd = rand.randint(1, 7)
    motion_blur_kernel_size = kernel_size_corrector(rand.randint(13, 30))
    motion_blur_angle = rand.uniform(0, 360)
    pixelation_magnitude = rand.randint(2, 4)
    perspective_pov = rand.uniform(-0.1, 0.1)
    perspective_bias = rand.uniform(0, 0.5)
    perspective_pov = perspective_pov - perspective_bias if perspective_pov < 0 else perspective_pov + perspective_bias
    resize_factor = rand.uniform(0.8, 1.00)
    contrast_level = rand.randint(1, 30)
    pixelation_magnitude = rand.randint(1, 4)

    seed = np.random.randint(99999)
    # Just rotate the original.
    original = deepcopy(img)
    #original = rs.apply_perspective(perspective_pov, original, seed)
    #original = rs.reduce_size(resize_factor, original, seed)
    #original = rs.random_border(original)

    # Rotate and corrupt the corrupted.
    #img = bl.gaussian_blur(gaussian_kernel_size, gaussian_sd, img)
    #img = bl.motion_blur(motion_blur_kernel_size, motion_blur_angle, img)
    #img = rs.apply_perspective(perspective_pov, img, seed)
    #img = rs.reduce_size(resize_factor, img, seed)
    #img = rs.random_border(img)
    #img = rs.random_gradient(img)
    #img = ct.increase_contrast(img, contrast_level)
    img = bl.pixelate_blur(pixelation_magnitude, img)

    return original, img

def nice_goal_image(img):
    # convert to np.float32
    Z = img.reshape((-1,3))
    Z = Z.astype(np.float32)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 10
    ret,label,center = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    return res2


def kernel_size_corrector(kernel_size):
    kernel_size = int(kernel_size)
    return kernel_size + 1 if kernel_size % 2 == 0 else kernel_size

if __name__ == "__main__":
    #the100 = input_data.load_images("data/100labeledLPforvalidation", 270, 90)
    image_data = input_data.load_images("data/40nicer", 270, 90)
    #input_, blurred = image_data.next_batch(50)
    # img = cv2.imread("data/40nice/0a0a7765-f5cc-4da9-b55f-d344e3fb2671-0.jpg")
    # img = corrupt(input_[0])
    #for i in range(10):
    #    cv2.imshow('Perspective', the100.imgs[i])
#        cv2.imshow('Ours', blurred[i])

    img = image_data.imgs[0]
    #mask = cv2.inRange(img, lower, upper)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([70,30,30])
    upper_blue = np.array([180,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    img[np.where((np.logical_and(hsv >= lower_blue, hsv <= upper_blue)).all(axis=2))] = [0,0,0]

    cv2.imshow('a', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

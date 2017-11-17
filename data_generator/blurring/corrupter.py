import data_generator.blurring.blurrer as bl
import data_generator.blurring.reshaper as rs
import data_generator.blurring.contrast as ct

import cv2
import numpy as np
import random as rand
from copy import deepcopy

def corrupt(img):
<<<<<<< HEAD
    gaussian_kernel_size = kernel_size_corrector(rand.randint(5, 9))
    gaussian_sd = rand.randint(3, 7)
    motion_blur_kernel_size = kernel_size_corrector(rand.randint(13, 21))
    motion_blur_angle = rand.uniform(0, 360)
    pixelation_magnitude = rand.randint(2, 4)
    perspective_pov = rand.uniform(-0.5, 0.5)
    resize_factor = rand.uniform(0.2, 1.00)
    contrast_level = rand.randint(20, 30)

    # Just rotate the original.
    original = nice_goal_image(deepcopy(img))
    #original = rs.apply_perspective(perspective_pov, original)
    #original = rs.reduce_size(resize_factor, original)
=======
    gaussian_kernel_size = kernel_size_corrector(rand.randint(3, 21))
    gaussian_sd = rand.randint(1, 7)
    motion_blur_kernel_size = kernel_size_corrector(rand.randint(5, 21))
    motion_blur_angle = rand.uniform(0, 360)
    pixelation_magnitude = rand.randint(2, 4)
    perspective_pov = rand.uniform(-0.4, 0.4)
    resize_factor = rand.uniform(0.8, 1.00)
    contrast_level = rand.randint(1, 30)
    pixelation_magnitude = rand.randint(1, 4)

    # Just rotate the original.
    original = deepcopy(img)
    original = rs.apply_perspective(perspective_pov, original)
    original = rs.reduce_size(resize_factor, original)
    #original = rs.random_border(original)
>>>>>>> 2b45d66530d16bf4b20d10d5799147952f837624

    # Rotate and corrupt the corrupted.
    img = bl.gaussian_blur(gaussian_kernel_size, gaussian_sd, img)
    img = bl.motion_blur(motion_blur_kernel_size, motion_blur_angle, img)
<<<<<<< HEAD
    #img = rs.apply_perspective(perspective_pov, img)
    #img = rs.reduce_size(resize_factor, img)
    img = ct.increase_contrast(img, contrast_level)
=======
    img = rs.apply_perspective(perspective_pov, img)
    img = rs.reduce_size(resize_factor, img)
    #img = rs.random_border(img)
    img = ct.increase_contrast(img, contrast_level)
    img = bl.pixelate_blur(pixelation_magnitude, img)
>>>>>>> 2b45d66530d16bf4b20d10d5799147952f837624

    return original, img

def nice_goal_image(img):
    # convert to np.float32
    Z = img.reshape((-1,3))
<<<<<<< HEAD

    Z = np.float32(Z)
=======
    Z = Z.astype(np.float32)
>>>>>>> 2b45d66530d16bf4b20d10d5799147952f837624

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
    image_data = input_data.load_images("data/40nice", 270, 90)
    input_, blurred = image_data.next_batch(1)
    # img = cv2.imread("data/40nice/0a0a7765-f5cc-4da9-b55f-d344e3fb2671-0.jpg")
    # img = corrupt(input_[0])

    cv2.imshow('Perspective', blurred[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

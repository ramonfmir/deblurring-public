import blurrer as bl
import reshaper as rs

import cv2
import random as rand

gaussian_kernel_size = 0
gaussian_sd = 0
motion_blur_kernel_size = 0
motion_blur_angle = 0
pixelation_magnitude = 0

perspective_pov = 0
resize_factor = 0
rotation_angle = 0

def corrupt(img):
    gaussian_kernel_size = 3
    gaussian_sd = 2
    motion_blur_kernel_size = kernel_size_corrector(rand.gauss(15, 1))
    motion_blur_angle = rand.gauss(90, 30)
    pixelation_magnitude = 3

    perspective_pov = rand.uniform(-0.5, 0.5)
    resize_factor = rand.uniform(0.5, 1.0)
    rotation_angle = rand.uniform(-5.0, 5.0)

    img = bl.gaussian_blur(gaussian_kernel_size, gaussian_sd, img)
    img = bl.pixelate_blur(pixelation_magnitude, img)
    img = bl.motion_blur(motion_blur_kernel_size, motion_blur_angle, img)
    img = rs.apply_perspective(perspective_pov, img)
    img = rs.reduce_size(resize_factor, img)
    img = rs.rotate_image(rotation_angle, img)
    img = bl.pixelate_blur(2, img)
    return img

def kernel_size_corrector(kernel_size):
    kernel_size = int(kernel_size)
    return kernel_size + 1 if kernel_size % 2 == 0 else kernel_size

if __name__ == "__main__":
    img = cv2.imread("../tests/license_plate.jpg")

    img = corrupt(img)

    cv2.imshow('Perspective', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import data_generator.blurring.blurrer as bl
import data_generator.blurring.reshaper as rs
import data_generator.blurring.contrast as ct

import cv2
import random as rand

def corrupt(img, corruption_rate=0.05):
    if rand.uniform(0, 1) < corruption_rate:
        return img

    # gaussian_kernel_size = 3
    # gaussian_sd = 2
    motion_blur_kernel_size = kernel_size_corrector(rand.randint(7, 17))
    motion_blur_angle = rand.uniform(0, 360)
    pixelation_magnitude = rand.randint(2, 4)
    contrast_level = rand.randint(10, 20)
    # perspective_pov = rand.uniform(-0.3, 0.3)
    # resize_factor = rand.uniform(0.75, 0.75)
    # rotation_angle = rand.uniform(-5.0, 5.0)

    # img = bl.gaussian_blur(gaussian_kernel_size, gaussian_sd, img)
    # img = bl.pixelate_blur(pixelation_magnitude, img)
    # img = rs.apply_perspective(perspective_pov, img) # done
    img = bl.motion_blur(motion_blur_kernel_size, motion_blur_angle, img)
    img = ct.increase_contrast(img, contrast_level)
    # img = rs.reduce_size(resize_factor, img)
    # img = rs.rotate_image(rotation_angle, img)
    # img = bl.pixelate_blur(pixelation_magnitude, img)
    return img

def kernel_size_corrector(kernel_size):
    kernel_size = int(kernel_size)
    return kernel_size + 1 if kernel_size % 2 == 0 else kernel_size

if __name__ == "__main__":
    img = cv2.imread("../tests/lp.jpg")

    img = corrupt(img)

    cv2.imshow('Perspective', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

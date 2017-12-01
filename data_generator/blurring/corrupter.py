import data_generator.blurring.blurrer as bl
import data_generator.blurring.reshaper as rs
import data_generator.blurring.contrast as ct
import data_generator.blurring.brightness_changer as bc

import cv2
import numpy as np
import random as rand

width = 270
height = 90

def blur_main(img):
    gaussian_kernel_size = kernel_size_corrector(rand.randint(3, 7))
    gaussian_sd = rand.randint(1, 5)
    motion_blur_kernel_size = kernel_size_corrector(rand.randint(11, 25))
    motion_blur_angle = rand.uniform(0, 360)
    perspective_pov = 0.5 - np.random.beta(0.5, 0.5)
    resize_factor = rand.uniform(0.8, 1.00)
    contrast_level = rand.randint(1, 20)
    pixelation_magnitude = rand.randint(4, 7)
    brightness_factor = rand.randint(0, 13)

    img, original = rs.reshape(resize_factor, perspective_pov, width, height, img)

    # Rotate and corrupt the corrupted.
    img = bl.gaussian_blur(gaussian_kernel_size, gaussian_sd, img)
    img = bl.motion_blur(motion_blur_kernel_size, motion_blur_angle, img)
    img = bc.change_brightness(brightness_factor, img)
    img = bl.pixelate_blur(pixelation_magnitude, img)
    img = ct.increase_contrast(img, contrast_level)

    return original, img

# Brightness
def blur_brightness(img):
    perspective_pov = rand.uniform(-0.4, 0.4)
    resize_factor = rand.uniform(0.7, 1.00)
    gaussian_kernel_size = kernel_size_corrector(rand.randint(3, 21))
    gaussian_sd = rand.randint(3, 9)
    motion_blur_kernel_size = kernel_size_corrector(rand.randint(3, 5))
    motion_blur_angle = rand.uniform(0, 360)

    img, original = rs.reshape(resize_factor, perspective_pov, width, height, img)

    # Rotate and corrupt the corrupted.
    img = bl.gaussian_blur(gaussian_kernel_size, gaussian_sd, img)
    img = bl.motion_blur(motion_blur_kernel_size, motion_blur_angle, img)
    img = bc.change_brightness(rand.randint(50, 100), img)

    return original, img

# Pixelation
def blur_pixelation(img):
    perspective_pov = rand.uniform(-0.4, 0.4)
    resize_factor = rand.uniform(0.7, 1.00)
    gaussian_kernel_size = kernel_size_corrector(rand.randint(9, 23))
    gaussian_sd = rand.randint(4, 7)
    motion_blur_kernel_size = kernel_size_corrector(rand.randint(3, 5))
    motion_blur_angle = rand.uniform(0, 360)
    pixelation_magnitude = rand.randint(4, 7)

    img, original = rs.reshape(resize_factor, perspective_pov, width, height, img)

    img = bl.gaussian_blur(gaussian_kernel_size, gaussian_sd, img)
    img = bl.motion_blur(motion_blur_kernel_size, motion_blur_angle, img)
    img = bl.pixelate_blur(pixelation_magnitude, img)


# Extreme vertical motion
def blur_vertical(img):
    perspective_pov = rand.uniform(-0.4, 0.4)
    resize_factor = rand.uniform(0.7, 1.00)
    gaussian_kernel_size = kernel_size_corrector(rand.randint(3, 9))
    gaussian_sd = rand.randint(1, 4)
    motion_blur_kernel_size = kernel_size_corrector(rand.randint(17, 29))
    motion_blur_angle = np.random.normal(90, 10) * (-1 if rand.randint(0, 1) == 0 else 1)

    img, original = rs.reshape(resize_factor, perspective_pov, width, height, img)

    img = bl.gaussian_blur(gaussian_kernel_size, gaussian_sd, img)
    img = bl.motion_blur(motion_blur_kernel_size, motion_blur_angle, img)

    return original, img

# Very mild blur
def blur_mild(img):
    perspective_pov = rand.uniform(-0.4, 0.4)
    resize_factor = rand.uniform(0.7, 1.00)
    gaussian_kernel_size = kernel_size_corrector(rand.randint(3, 5))
    gaussian_sd = rand.randint(1, 2)
    motion_blur_kernel_size = kernel_size_corrector(rand.randint(3, 5))
    motion_blur_angle = rand.uniform(0, 360)
    brightness_factor = rand.randint(0, 5)

    img, original = rs.reshape(resize_factor, perspective_pov, width, height, img)

    img = bl.gaussian_blur(gaussian_kernel_size, gaussian_sd, img)
    img = bl.motion_blur(motion_blur_kernel_size, motion_blur_angle, img)
    img = bc.change_brightness(brightness_factor, img)

    return original, img

blurs = [blur_main, blur_brightness, blur_pixelation, blur_vertical, blur_mild]
blur_b_freq = 3
blur_p_freq = 2
blur_v_freq = 5
blur_m_freq = 2

def corrupt(img):
    blur_selector = 0

    random = rand.randint(1, 100)
    if random <= blur_b_freq:
        blur_selector = 1 # blur_brightness
    elif random <= blur_b_freq + blur_p_freq:
        blur_selector = 2 # blur_pixelation
    elif random <= blur_b_freq + blur_p_freq + blur_v_freq:
        blur_selector = 3 # blur_vertical
    elif random <= blur_b_freq + blur_p_freq + blur_v_freq + blur_m_freq:
        blur_selector = 4 # blur_mild

    return blurs[blur_selector](img)

def kernel_size_corrector(kernel_size):
    kernel_size = int(kernel_size)
    return kernel_size + 1 if kernel_size % 2 == 0 else kernel_size

if __name__ == "__main__":
    #image_data = input_data.load_images("data/40nice", 270, 90)
    #input_, blurred = image_data.next_batch(1)
    img = cv2.imread("data/3548awesome/chinahainan_29.jpg")
    cv2.imshow('Good', img)

    cv2.waitKey(0)

    for i in range(100):
        img = cv2.imread("data/3548awesome/chinahainan_29.jpg")

        cv2.imshow('Bad', blur_main(img)[1])
        cv2.waitKey(0)

    cv2.destroyAllWindows()

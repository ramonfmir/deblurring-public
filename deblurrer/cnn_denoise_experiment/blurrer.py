from skimage import color
import numpy as np
from scipy.ndimage.filters import gaussian_filter

def motion_blur(img):
    return img

def sensor_blur(img):
    return img

def optical_blur(img):
    filtered_gray = gaussian_filter(color.rgb2gray(img), sigma = 5)
    return color.gray2rgb(filtered_gray)

def blur_all(imgs):
    return np.asarray([optical_blur(img) for img in imgs])


def add_noise(img):
    noise_mask = np.random.binomial(1, 1 - corruption_level, img.shape)
    return noise_mask * img

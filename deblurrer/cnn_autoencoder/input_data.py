import sys
sys.path.append("../..")
import cv2
import os
import glob
import numpy as np
import random
from skimage import color
import blurrer.blurring_experiment as blurrer
from functools import partial
# from ..blurrer import blurring_experiment

# loads the image from file into array
# The unziped files of images must exits in the relative directory
# ../datasets/4000unlabeledLP_same_dims_scaled

def load_images(train_path, image_size_x,image_size_y):
    images = []
    img_names = []
    path = os.path.join(train_path, '*g')
    # Get all files in the directory
    files = glob.glob(path)
    print('Now going to read files {}'.format(path))
    for fl in files:
        image = cv2.imread(fl)
        image = cv2.resize(image, (image_size_x, image_size_y),0,0, cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        # Normalise colour
        image = np.multiply(image, 1.0 / 255.0)
        images.append(image)
        flbase = os.path.basename(fl)
        img_names.append(flbase)
    random.shuffle(images)
    # images: List of images in Array form;
    # img_names: The list of corresponding image file name;
    return data_set(images)

class data_set(object):
    def __init__(self, imgs):
        self.imgs = imgs
        self.train_set_pointer = 0
        kernel_size = 5
        averaging_b = partial(blurrer.averaging_blur, kernel_size)
        median_b    = partial(blurrer.median_blur, kernel_size)
        gaussian_b  = partial(blurrer.gaussian_blur, kernel_size, 0.05)
        bilateral_b = partial(blurrer.bilateral_blur, kernel_size,  10)
        motion_b    = partial(blurrer.motion_blur, kernel_size, 'H')

        self.blur_func_set = [averaging_b, median_b, gaussian_b, bilateral_b, motion_b]


    # next_batch retunr tuple of unblurred image and corrupted image with set
    # blurring parameters
    def next_batch(self, batch_size):
        batch_start_index = self.train_set_pointer
        batch_end_index = batch_start_index + batch_size
        if (batch_end_index> len(self.imgs)):
            batch_start_index = 0
            batch_end_index = batch_start_index + batch_size
            self.train_set_pointer = 0
        else:
            self.train_set_pointer = batch_end_index
        batch = self.imgs[batch_start_index:batch_end_index]
        if self.train_set_pointer == 0:
            random.shuffle(self.imgs)
        return np.asarray(batch), np.asarray(self.blur_data_set(batch))

    def blur_data_set(self, original_batch):
        #TODO
        #More predicatble blur data, move blur set generation to blurrer.py
        corrupted = [blurrer.apply_blurs_randomly(self.blur_func_set, img) for img in original_batch]
        return corrupted

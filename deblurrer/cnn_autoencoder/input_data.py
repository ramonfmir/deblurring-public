import sys
sys.path.insert(0, "../..")
import cv2
import os
import glob
import numpy as np
np.set_printoptions(threshold=np.inf)
import scipy.misc as sc
import random
from skimage import color
from functools import partial
from blurrer import blurring_experiment as blurrer

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
        bilateral_b = partial(blurrer.bilateral_blur, kernel_size,  10)
        motion_b_H    = partial(blurrer.motion_blur, 14, 'H', 2)
        motion_b_V    = partial(blurrer.motion_blur, 9, 'V', 2)

        self.blur_func_set = [motion_b_H,motion_b_V, bilateral_b]

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

    """ create motion blur kernel. lens = strength of blur, theta = angle of blur
    """
    def motion_blur_kernel(self,lens,theta):
        # https://sourceforge.net/p/octave/image/ci/default/tree/inst/fspecial.m#l379
        if(lens < 1):
            lens = 9
        if (np.mod(lens, 2) == 1):
            sze = [lens, lens]
        else:
            sze = [lens+1, lens+1]
        ## First generate a horizontal line across the middle
        f = np.zeros(sze)
        f[int(np.floor(lens/2))][0:lens] = 1
        # Then rotate to specified angle
        f = sc.imrotate(f,theta)
        f = f / np.sum(f);
        return f

    def blur_data_set(self, original_batch):
        kernel = self.motion_blur_kernel(9,20)
        corrupted = [cv2.filter2D(img,-1,kernel) for img in original_batch]
        return corrupted

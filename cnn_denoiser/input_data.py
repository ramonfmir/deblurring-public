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
from data_generator.blurring import corrupter

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
        image = cv2.resize(image, (image_size_x, image_size_y), 0, 0, cv2.INTER_CUBIC)
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

        return self.training_batch(batch)#np.asarray(batch), np.asarray(self.blur_batch(batch))

    def training_batch(self, original_batch):
        goal_batch = []
        corrupted_batch = []
        for img in original_batch:
            goal, corrupted = corrupter.corrupt(img)
            goal_batch.append(goal)
            corrupted_batch.append(corrupted)

        return goal_batch, corrupted_batch


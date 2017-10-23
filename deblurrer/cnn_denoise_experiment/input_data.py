import sys
import cv2
import os
import glob
import numpy as np
import random

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
    return data_set(images, 0.99)

class data_set(object):
    def __init__(self, imgs, train_set_perc):
        self.imgs = imgs
        self.train_set_perc = train_set_perc
        self.train_set_pointer = 0

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
        print(np.asarray(batch).shape)
        return np.asarray(batch)

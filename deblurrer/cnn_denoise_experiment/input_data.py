import sys
import cv2
import os
import glob
import numpy as np

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
    images = np.array(images)
    img_names = np.array(img_names)
    # images: List of images in Array form;
    # img_names: The list of corresponding image file name;
    return images, img_names

import glob, os

import numpy as np
import cv2
import sys

from copy import deepcopy
from itertools import permutations


char_ratio = 0.129
start_ratio = 0.024
dot_ratio = 0.0455

def get_n_char(img_src, n):
    img = cv2.imread(img_src)
    height, width, _  = img.shape

    start = int(width*start_ratio)
    char_size = int(width*char_ratio)

    if (n < 2):
        return img[ : , start + n*char_size : start + (n+1)*char_size ]
    else:
        dot_size = int(width*dot_ratio)
        return  img[ : , start + dot_size + n*char_size : start + dot_size + (n+1)*char_size ]

def permute(img_src):
    img = cv2.imread(img_src)

    chars_perms = []
    for i in range(1,7):
        char = get_n_char(img_src, i)
        chars_perms.append(char)

    chars_perms = list(permutations(chars_perms))
    height, width, _  = img.shape

    start = int(width*start_ratio)
    char_size = int(width*char_ratio)
    dot_size = int(width*dot_ratio)

    permed_image = deepcopy(img)

    for i in range(1, len(chars_perms)):
        for n, chars in enumerate(chars_perms[i]):
            n += 1
            if (n < 2):
                permed_image[ : , start + n*char_size : start + (n+1)*char_size ] = chars
            else:
                permed_image[ : , start + dot_size + n*char_size : start + dot_size + (n+1)*char_size ] = chars
        img_dst = img_src + '_' + str(i)
        print("writing to ", img_dst)
        cv2.imwrite(img_dst, permed_image)

    # return permed_image

if __name__ == '__main__':
    dataset_path = sys.argv[1]
    for file_path in os.listdir(dataset_path):
        file_path = dataset_path + "/" + file_path
        permute(file_path)

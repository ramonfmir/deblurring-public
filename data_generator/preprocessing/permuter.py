import glob, os

import numpy as np
import random as rand
import cv2
import sys

from copy import deepcopy
from itertools import permutations


char_ratio = 0.116
start_ratio = 0.034
dot_ratio = 0.0460
char_space = 0.014

letter_positions = [start_ratio]
pos = start_ratio
for i in range(1):
    pos += char_ratio + char_space
    letter_positions.append(pos)

pos += dot_ratio

for i in range(5):
    pos += char_ratio + char_space
    letter_positions.append(pos)


def get_n_char(img_src, n):
    img = cv2.imread(img_src)
    height, width, _  = img.shape

    start = int(width*start_ratio)
    char_size = int(width*char_ratio)
    space = int(max((n - 1) * char_space, 0))
    if (n < 2):
        return img[ : , int(width * letter_positions[n]) : int(width * letter_positions[n]) + int(width * char_ratio) ]
    else:
        dot_size = int(width*dot_ratio)
        return img[ : , int(width * letter_positions[n]) : int(width * letter_positions[n]) + int(width * char_ratio)]

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

    for i in range(50):
        for n, chars in enumerate(chars_perms[rand.randint(0, len(chars_perms) - 1)]):
            n += 1
            space = int(max((n - 1) * char_space, 0))
            if (n < 2):
                permed_image[ : , int(width * letter_positions[n]) : int(width * letter_positions[n]) + int(width * char_ratio)] = chars
            else:
                permed_image[ : , int(width * letter_positions[n]) : int(width * letter_positions[n]) + int(width * char_ratio)] = chars
        img_path, img_extension = filename, file_extension = os.path.splitext(img_src)
        img_dst = img_path + '_' + str(i) + img_extension
        print("writing to ", img_dst)
        cv2.imwrite(img_dst, permed_image)

    # return permed_image

if __name__ == '__main__':
    dataset_path = sys.argv[1]
    for file_path in os.listdir(dataset_path):
        file_path = dataset_path + "/" + file_path
        # for i in range(7):
        #     char = get_n_char(file_path, i)
        #     cv2.imshow('faaa', char)
        #     cv2.waitKey(0)
        permute(file_path)

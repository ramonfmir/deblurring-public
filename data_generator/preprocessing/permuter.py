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

def permute(img, chars_perms):
    height, width, _  = img.shape

    start = int(width*start_ratio)
    char_size = int(width*char_ratio)
    dot_size = int(width*dot_ratio)

    permed_image = deepcopy(img)

    for n, chars in enumerate(chars_perms[4800]):
        if (n < 2):
            permed_image[ : , start + n*char_size : start + (n+1)*char_size ] = chars
        else:
            permed_image[ : , start + dot_size + n*char_size : start + dot_size + (n+1)*char_size ] = chars

    return permed_image

if __name__ == '__main__':
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    chars = []
    for i in range(7):
        char = get_n_char(img_path, i)
        chars.append(char)

    permed_chars = list(permutations(chars))
    test = permute(img, permed_chars)

    cv2.imshow("char", test)
    cv2.waitKey(0)

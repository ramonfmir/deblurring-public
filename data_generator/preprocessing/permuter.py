import glob, os

import numpy as np
import random as rand
import cv2
import sys
import numpy as np
import data_generator.preprocessing.colour_normalize as cn

from copy import deepcopy
from itertools import permutations

base = cv2.imread("../../data/hackedIms/baseNorm.jpg")

char_ratio = 0.116
start_ratio = 0.034
dot_ratio = 0.0460
char_space = 0.014
top_ratio = 0.131

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
        return img[ : , int(width * letter_positions[n]) : int(width * letter_positions[n]) + int(width * char_ratio)]
    else:
        dot_size = int(width*dot_ratio)
        return img[ : , int(width * letter_positions[n]) : int(width * letter_positions[n]) + int(width * char_ratio)]

def put_n_char(img, n, char):
    height, width, _  = img.shape
    start = int(width*start_ratio)
    char_size = int(width*char_ratio)
    top_crop = int(height * top_ratio)
    space = int(max((n - 1) * char_space, 0))
    if (n < 2):
        img[top_crop :-top_crop, int(width * letter_positions[n]) : int(width * letter_positions[n]) + int(width * char_ratio)] = char[top_crop: -top_crop, :]
    else:
        dot_size = int(width*dot_ratio)
        img[top_crop :-top_crop, int(width * letter_positions[n]) : int(width * letter_positions[n]) + int(width * char_ratio)] = char[top_crop: -top_crop, :]
    return img

def the_biggest_of_hacks():
    path_truths = []
    while (True):
        try:
            in_ = input().split(' ')
            path, ground_truth = in_[0], in_[1]
            if not ('*' in ground_truth or '#' in ground_truth):
                path_truths.append((path, ground_truth))
        except EOFError:
            print("All plates processed")
            break

    height, width, _ = base.shape
    char_width = int(width * char_ratio)
    dst_folder = os.path.abspath("../../data/generated/")
    for path, truth in path_truths:
        generated_img = deepcopy(base)
        for i, character in enumerate(truth):
            char_path = "../../data/hackedIms/all/%s.jpg" % character
            if (not os.path.exists(char_path)):
                continue
            char_img = cv2.imread(os.path.abspath("../../data/hackedIms/all/%s.jpg" % character))
            char_img = cv2.resize(char_img, (char_width, height), 0, 0, cv2.INTER_CUBIC)
            generated_img = put_n_char(generated_img, i, char_img)

        cv2.imwrite(dst_folder + "/" + path, generated_img)

def gen_data():
    letters = "0123456789abcdehijklmnpqrstuvwxyz京浙豫津鄂沪辽"
    dst_folder = os.path.abspath("../../data/hackedIms/uniform_yellow/")
    length = len(letters)
    height, width, _ = base.shape
    char_width = int(width * char_ratio)
    N = 7
    GEN_SIZE = 1000

    for j in range(GEN_SIZE):
        print(j)
        vals = np.random.randint(0,length,(N)).tolist()
        generated_img = deepcopy(base)
        for i in range(N):
            char_path = "../../data/hackedIms/all/%s.jpg" % letters[vals[i]]
            char_img = cv2.imread(os.path.abspath(char_path))
            char_img = cv2.resize(char_img, (char_width, height), 0, 0, cv2.INTER_CUBIC)
            generated_img = put_n_char(generated_img, i, char_img)
        cv2.imwrite(dst_folder + "/" + str(j) + ".jpg", generated_img)

    cn.normalize("../../data/hackedIms/uniform_yellow/")

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

        cv2.imwrite(img_dst, permed_image)

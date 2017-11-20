import sys
import numpy as np
import cv2
import random as rand

def increase_brightness(amount, img):
    img = np.where((255 - img) < amount, 255, img + amount)
    return img

def decrease_brightness(amount, img):
    img = np.where(img < amount, 0, img - amount)
    return img

def change_brightness(amount, img):
    if rand.randint(0, 1) == 0:
        return decrease_brightness(amount, img)
    else:
        return increase_brightness(amount, img)

if __name__ == "__main__":
    #image_data = input_data.load_images("data/40nice", 270, 90)
    #input_, blurred = image_data.next_batch(1)
    img = cv2.imread("data/40nice/0a0a7765-f5cc-4da9-b55f-d344e3fb2671-0.jpg")
    cv2.imshow('Good', img)
    cv2.waitKey(0)

    cv2.imshow('Bad', decrease_brightness(150, img))
    cv2.waitKey(0)

    cv2.destroyAllWindows()
import numpy as np
import cv2
import sys
import os
import glob

lower_blue = np.array([110,50,50], dtype=np.uint8)
upper_blue = np.array([130,255,255], dtype=np.uint8)

lower_yellow = np.array([25,50,50], dtype=np.uint8)
upper_yellow = np.array([32,255,255], dtype=np.uint8)

lower_black = np.array([0, 0, 0], dtype=np.uint8)
upper_black = np.array([180, 255, 30], dtype=np.uint8)

lower_white = np.array([0, 0, 200], dtype=np.uint8)
upper_white = np.array([180, 255, 255], dtype=np.uint8)

def is_blue(img):
    color_dict = {'blue':0, 'yellow':0, 'white':0, 'black':0}
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for row in img:
        for pixel in row:
            if (pixel > lower_blue).all() and (pixel < upper_blue).all():
                color_dict['blue'] = color_dict['blue']+1
            elif (pixel > lower_yellow).all() and (pixel < upper_yellow).all():
                color_dict['yellow'] = color_dict['yellow']+1
            elif (pixel > lower_white).all() and (pixel < upper_white).all():
                color_dict['white'] = color_dict['white']+1
            elif (pixel > lower_black).all() and (pixel < upper_black).all():
                color_dict['black'] = color_dict['black']+1
    # yb_contrib = color_dict['yellow'] + color_dict['black']
    # bw_contrib = color_dict['blue'] + color_dict['white']
    return color_dict['blue']> color_dict['yellow']



if __name__ == '__main__':
    imgs_path = sys.argv[1]

    # Get all files in the directory
    path = os.path.join(imgs_path, '*g')
    files = glob.glob(path)

    # Output quantized images into out_path
    for fl in files:
        img = cv2.imread(fl)
        print ("is blue ", is_blue(img))
        cv2.imshow('LP', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

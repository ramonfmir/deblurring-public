import numpy as np
import cv2
import sys
import glob
import os

def knn_colour(img_src, img_dst, K):
    img = cv2.imread(img_src)
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # K = 2
    attempts = 30
    ret,label,center=cv2.kmeans(Z,K,None,criteria,attempts,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    print("writing to ", img_dst)
    cv2.imwrite(img_dst, res2)

if __name__ == '__main__':
    imgs_path = sys.argv[1]
    out_path  = sys.argv[2]
    K         = int(sys.argv[3])

    # Get all files in the directory
    path = os.path.join(imgs_path, '*g')
    files = glob.glob(path)

    # Ensure output directory exists
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Output quantized images into out_path
    count = 0.0
    size  = len(files)
    print(size)
    for fl in files:
        print(count / size)
        img_dst = os.path.join(out_path, os.path.basename(fl))
        knn_colour(fl, img_dst, K)
        count += 1

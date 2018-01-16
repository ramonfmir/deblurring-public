import numpy as np
import cv2
import sys
import glob
import os

def knn_colour(img_src, img_dst, K):
    img = cv2.imread(img_src)
    Z = img.reshape((-1,3))

    # Convert to np.float32
    Z = np.float32(Z)

    # Define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # K = 2
    attempts = 30
    ret,label,center=cv2.kmeans(Z,K,None,criteria,attempts,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    cv2.imwrite(img_dst, res2)

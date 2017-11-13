import sys
import numpy as np
import cv2

def remove_rotation(img):
    Z = img.reshape((-1,3))

    # convert to np.float32
    #Z = img

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
    K = 8
    ret,label,center = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image

    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    return res2

if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])
    img = img.astype(np.float32)
    img = np.multiply(img, 1.0 / 255.0)

    cv2.imshow('Rotation', img)
    cv2.waitKey(0)

    img = remove_rotation(img)

    cv2.imshow('No rotation', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

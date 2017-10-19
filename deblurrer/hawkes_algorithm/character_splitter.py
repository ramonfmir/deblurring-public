import sys

import numpy as np
import cv2

im = cv2.imread('../../data/4000unlabeledLP/00a5b119-a673-4800-b61f-dd4e64cd5364-0.jpg')
#im = cv2.imread('numbers.png')
im3 = im.copy()
"""
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(7,7),0)
blur = cv2.medianBlur(blur, 5)
size = 10
kernel_motion_blur = np.zeros((size, size))
kernel_motion_blur[:, int((size - 1) / 2)] = np.ones(size)
kernel_motion_blur = kernel_motion_blur / size
blur = cv2.filter2D(blur, -1, kernel_motion_blur)
"""
blur = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

cv2.imshow('norm', blur)
cv2.waitKey(0)

w,h = blur.shape
print str(w) + " " + str(h)
pts1 = np.float32([[0,h],[w,h], [0,0],[w,0]])
pts2 = np.float32([[6,24],[66,31], [8,1],[68,12]])
M = cv2.getPerspectiveTransform(pts1,pts2)
blur = cv2.warpPerspective(blur,M,(6,68))

#thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

cv2.imshow('norm', blur)
cv2.waitKey(0)

#################      Now finding Contours         ###################
"""
_,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

samples =  np.empty((0,100))
responses = []
keys = [i for i in range(48,58)]

for cnt in contours:
    if cv2.contourArea(cnt)>5:
        [x,y,w,h] = cv2.boundingRect(cnt)

        if  h > 16 and h > w:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            cv2.imshow('norm',im)
            key = cv2.waitKey(0)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1,100))
                samples = np.append(samples,sample,0)

responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print "training complete"

np.savetxt('generalsamples.data',samples)
np.savetxt('generalresponses.data',responses)
"""
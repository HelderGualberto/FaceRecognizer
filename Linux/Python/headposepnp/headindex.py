#!/usr/bin/env python
 
import cv2
import numpy as np
from align_dlib import AlignDlib
from os.path import join
 
# Read Image
im = cv2.imread("headl.png");
size = im.shape
align = AlignDlib(join("..","..","data","models","dlib","shape_predictor_68_face_landmarks.dat"))
bb = align.getLargestFaceBoundingBox(im)
if bb is not None:
    landmarks = align.findLandmarks(im, bb)

for idx,p in enumerate(landmarks):
    cv2.circle(im, p , 3, (0,0,255), -1)
    cv2.putText(im,str(idx),p,cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),1)

 
# Display image
cv2.imshow("Output", im)
cv2.waitKey(0)
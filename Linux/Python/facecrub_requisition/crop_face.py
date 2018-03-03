import cv2
import urllib
import numpy as np

req = open("/home/helder/usp/images/teste.jpg","rw")
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
img = cv2.imdecode(arr,-1) # 'load it as it is'


cv2.imwrite("foto.jpg",img)

cv2.imshow('lalala',img)
cv2.waitKey(0)
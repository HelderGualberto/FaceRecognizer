import cv2, os
import numpy as np

def read_images_from_directory(path):
	files = os.listdir(path)
	imgs = []

	for name in files:
		filename = path + name

		mat = cv2.imread(filename)
		imgs.append(mat)

	print files
	return imgs

def correct_gamma(gamma, mat):

	tmp = mat.copy()
	(rows, cols, c) = tmp.shape

	for i in range(rows):
		for j in range(cols):
			for k in range(c):
				normRGB = tmp.item(i,j,k)/255.0
				gammaPower = np.power(normRGB,(1.0/float(gamma)))
				RGB = gammaPower*255.0
				tmp.itemset(i,j,k,RGB)
	return tmp

input_path = '../data/input/'

output_path = '../data/output/'

imgs = read_images_from_directory(input_path)


cv2.imshow('org', imgs[0])
mat_gamma = correct_gamma(2.0, imgs[0])
# cv2.imshow('1.1-1', mat_gamma)
# mat_gamma = correct_gamma(2.0, mat_gamma)
# cv2.imshow('1.1-2', mat_gamma)
# mat_gamma = correct_gamma(2.0, mat_gamma) 
cv2.imshow('1.1-3', mat_gamma)
cv2.imwrite(output_path+'img.jpg', mat_gamma)

cv2.waitKey(0)



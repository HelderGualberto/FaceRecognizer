import cv2, os
from cv2 import cv
import numpy as np
import argparse
from matplotlib import pyplot as plt


def read_images_from_directory(path):
	files = os.listdir(path)
	imgs = []

	for name in files:
		filename = path + name

		mat = cv2.imread(filename)
		imgs.append(mat)

	print files
	return imgs, files

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** (invGamma)) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def log_gamma_ajust(image):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values

	table = np.array([((i / 255.0) ** (2.10380372/np.log10(i+2))) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


def correlation(mat, limit=50):

	histogram = cv2.calcHist([mat],[0],None,[256],[0,256])

	lighter = histogram[limit:]
	darker = histogram[0:limit]

	sum_light = np.sum(lighter)
	sum_dark = np.sum(darker)

	return sum_dark/sum_light		


def rotate_img(mat, angle=270):
	rows,cols,c = mat.shape
	M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
	dst = cv2.warpAffine(mat,M,(cols,rows))
	return dst

parser = argparse.ArgumentParser()

parser.add_argument('--video', type=str, help="URL do video. Se nao indicado faz a leitura das imagens na pasta especificada.", default=None)

input_path = '../data/input/'
output_path = '../data/output/'

args = parser.parse_args()

c = 0
if args.video != None:
	cap = cv2.VideoCapture(args.video)

	ret, frame = cap.read()

	while ret:
		ret, frame = cap.read()
		#frame = rotate_img(frame)

		# if correlation(frame) > 4.0:
		mat_gamma = log_gamma_ajust(frame)
		# else:
		# 	mat_gamma = frame

		#histGamma = cv2.calcHist([mat_gamma],[0],None,[256],[0,256])
		cv2.imshow('gamma', mat_gamma)
		cv2.imshow('org', frame)
		cv2.waitKey(0)

		# plt.figure(1)
		# plt.plot(hist, label="Original")
		# plt.figure(2)
		# plt.plot(histGamma, label="Gamma 2.0")
		# plt.show()

		imname = output_path + 'img_' + str(c) + '.jpg'
		print "Saving: {}".format(imname)
		cv2.imwrite(imname,frame)
		c+=1


		#cv2.imshow('gamma',mat_gamma)
		#cv2.waitKey(30)


else:
	imgs, filename = read_images_from_directory(input_path)

	#rng = np.arange(1.0,5.1,0.1,dtype=np.float)

	rng = [2.2]


	for gamma in rng:
		print "Calculating Gamma " + str(gamma)
		

		for img in imgs:
			plt.legend()
			output = output_path + str(gamma)

			if not os.path.exists(output):
				os.makedirs(output)

			#plt.plot(hist, label=filename[c])
			
			mat_gamma = adjust_gamma(img, gamma=2.2)

			mat_log_gamma = log_gamma_ajust(mat_gamma)

			gray = cv2.cvtColor(mat_log_gamma, cv2.COLOR_BGR2GRAY)

			imname = output + '/img_log' + str(c) + '.jpg'
			cv2.imwrite(imname,mat_log_gamma)

			
			gray = cv2.equalizeHist(gray)

			imname = output + '/img_gamma' + str(c) + '.jpg'
			cv2.imwrite(imname,mat_gamma)

			c+=1
		# plt.legend()
		# plt.show()
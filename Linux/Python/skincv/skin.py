# import the necessary packages
#from pyimagesearch 
import imutils
import numpy as np
import argparse
import cv2
#para plotar imagem do histograma
from matplotlib import pyplot as plt
import copy 
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help = "path to the (optional) video file")
args = vars(ap.parse_args())
#carregando cascasde de rosto
face_cascade = cv2.CascadeClassifier('D:\projetos\Safety_City\Code\data\haarcascades\haarcascade_frontalface_default.xml')
# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")
# if a video path was not supplied, grab the reference
# to the gray
if not args.get("video", False):
	camera = cv2.VideoCapture(0)
 
# otherwise, load the video
else:
	camera = cv2.VideoCapture(args["video"])
	# keep looping over the frames in the video
	while True:
		# grab the current frame
		(grabbed, frame) = camera.read()
 
		# if we are viewing a video and we did not grab a
		# frame, then we have reached the end of the video
		if args.get("video") and not grabbed:
			break
 
 		# obtendo a imagem em escala de cinza
 		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 		#detectando as faces na imagem
 		faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(120, 120),flags=cv2.CASCADE_SCALE_IMAGE)
 		img_sw = copy.copy(frame)
 		facecolor = [(255,0,0),(0,255,0),(0,0,255),(128,0,0),(0,128,0),(0,0,128)]
 		idxc = 0
 		for (x,y,w,h) in faces:
 			# recorte da area de interesse da imagem
 			roi_img = frame[(y+(h/10)):y+h-h/2, (x+w/3):x+w-(w/3)]
 			cv2.rectangle(img_sw,(x,y),(x+w,y+h),facecolor[idxc % 6],2) 			
 			#
 			#histograma rgb
 			#color = ('b','g','r')
 			#color = ('h','s','v')
 			#roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
 			#for i,col in enumerate(color):
 			#	histr = cv2.calcHist([roi_img],[i],None,[256],[0,256])
 			#	plt.plot(histr,color = col)
 			#	plt.xlim([0,256])
 			#plt.show()
 			#cv2.imshow('face',roi_img)
 			#cv2.waitKey(0)
 			hsv = cv2.cvtColor(roi_img,cv2.COLOR_BGR2HSV)
 			target = copy.copy(frame)
 			hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
 			# calculating object histogram
 			roihist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
 			# normalize histogram and apply backprojection
 			cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
 			dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
 			# Now convolute with circular disc
 			disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
 			cv2.filter2D(dst,-1,disc,dst)
 			# threshold and binary AND
 			ret,thresh = cv2.threshold(dst,50,255,0)
 			thresh = cv2.merge((thresh,thresh,thresh))
 			res = cv2.bitwise_and(target,thresh)
 			cv2.imshow('mascara',thresh)
 			cv2.rectangle(res,(x,y),(x+w,y+h),facecolor[idxc % 6],2)
 			idxc += 1
 			cv2.imshow('Resultado',res)
 			if cv2.waitKey(500) & 0xFF == ord("q"):
 				break
 		cv2.imshow('faces detectadas',img_sw)
 		
		# resize the frame, convert it to the HSV color space,
		# and determine the HSV pixel intensities that fall into
		# the speicifed upper and lower boundaries
		
		frame = imutils.resize(frame, width = 400)
		converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		skinMask = cv2.inRange(converted, lower, upper)
 
		# apply a series of erosions and dilations to the mask
		# using an elliptical kernel
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
		skinMask = cv2.erode(skinMask, kernel, iterations = 2)
		skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
 
		# blur the mask to help remove noise, then apply the
		# mask to the frame
		skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
		skin = cv2.bitwise_and(frame, frame, mask = skinMask)
 
		# show the skin in the image along with the mask
		cv2.imshow("images", np.hstack([frame, skin]))
 
		# if the 'q' key is pressed, stop the loop
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break
 
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

	

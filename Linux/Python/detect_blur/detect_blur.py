# import the necessary packages
from imutils import paths
import argparse
import cv2
import numpy as np
from os.path import join
from align_dlib import AlignDlib
from matplotlib import pyplot as plt

align = AlignDlib(join("..","..","data","models","dlib","shape_predictor_68_face_landmarks.dat"))
 
def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

def extratctFaceAli(sample):
	bb = align.getLargestFaceBoundingBox(sample)
	#processa para separar somente rosto mesmo extraido de haar
	if bb is None:
		print "Face not found in dlib"
		return False,None,0
	else:
		samplec = sample[bb.top():bb.bottom(),bb.left():bb.right()].copy()
		h,w,c = samplec.shape
		if h < 20 or w < 20 :
			print "Strange format in dlib"
			return False,None,0
		md = w if w > h else h
		if md > 96:
			fator = 96.0/float(md)
			samplec = cv2.resize(samplec,(0,0),fx=fator,fy=fator)
			hr,wr,cr = samplec.shape
			if hr < 20 or wr < 20:
				print "Error after resize prev {}x{} after {}x{} orig:{}".format(h,w,hr,wr,sample.shape)
				return False,None,0
		#landmarks = align.findLandmarks(sample, bb)
		#sample = align.align(96, sample, bb,
		#			landmarks=landmarks,
		#			landmarkIndices=align.OUTER_EYES_AND_NOSE)
		return True,samplec,w

# mouse callback function
def processa(event,x,y,flags,param):
   if event == cv2.EVENT_LBUTTONDBLCLK:
    print event,x,y,flags,param

def extractMagFFT(img):
	f = np.fft.fft2(img)
	fshift = np.fft.fftshift(f)
	#magimg = np.log(np.abs(fshift)) #20*
	magimg=np.abs(fshift)
	return magimg

# construct the argument parse and parse the arguments
#required=True,
ap = argparse.ArgumentParser()
#D:\\projetos\\Safety_City\\Code\data\\Rev\\predicpng
#D:\\projetos\\Safety_City\\Bases\\usp\\filtrada_up160802\\database_com_dlibfilter_02-08-2016'
#D:\\projetos\\Safety_City\\Code\\data\\video_camara_02\\rawcrowreps2
#D:\\projetos\\Safety_City\\Bases\\local\\ajusta\\localhighrpre
#D:\\projetos\\Safety_City\\Bases\\local\\localhighpre
#D:\\projetos\\Safety_City\\Bases\\compara\\peoplepre
ap.add_argument("-i", "--images", 
			#default='D:\\projetos\\Safety_City\\Bases\\local\\ajusta\\localhighrpre',
			#default="D:\\Safety_City\\imagens\\teste",
			#default="D:\Safety_City_codep\usputil\img",
			#default="D:\\Safety_City\\imagens\\qualityimg",
			#default="D:\\Safety_City\\imagens\\camcorredor8mb",
			#default="D:\\Safety_City\\imagens\\p_1222084100",
			default="D:\\Safety_City\\imagens\\amostrab",
	help="path to input directory of images")
#ap.add_argument("-t", "--threshold", type=float, default=90.0,
#ap.add_argument("-t", "--threshold", type=float, default=100.0,
ap.add_argument("-t", "--threshold", type=float, default=125.0,
	help="focus measures that fall below this value will be considered 'blurry'")
#ap.add_argument("-f", "--threshold2", type=float, default=0.10,
ap.add_argument("-f", "--threshold2", type=float, default=0.125,
#ap.add_argument("-f", "--threshold2", type=float, default=0.150,
	help="focus measures that fall below this value will be considered 'blurry' in fft")
args = vars(ap.parse_args())
# loop over the input images
largeimg = np.zeros((960,1440,3), np.uint8)
largeimgGray = np.zeros((960,1440), np.uint8)

xpos=0
ypos=0
conta = 0

cv2.namedWindow('Image')
cv2.setMouseCallback('Image',processa,param = 'valor')

for imagePath in paths.list_images(args["images"]):
	# load the image, convert it to grayscale, and compute the
	# focus measure of the image using the Variance of Laplacian
	# method
	image = cv2.imread(imagePath)
	partpp = imagePath.split('\\')
	if image is None:
		continue
	copia = image.copy()
	ret,image,wori=extratctFaceAli(image)
	if not ret:
		cv2.imshow("{} Error!!".format(partpp[len(partpp)-1]),copia)
		k=cv2.waitKey(0)
		continue
	h,w,c = image.shape
	if h == 0 or w == 0:
		continue
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#gray = cv2.equalizeHist(gray)
	snapg = gray #[0:48,12:84]
	
	#plt.subplot(121),plt.imshow(gray, cmap = 'gray')
	#plt.title('Input Image'), plt.xticks([]), plt.yticks([])	
	
	magimg = extractMagFFT(gray)
	maxv = np.amax(magimg)
	#print "Max DC FFT:{}".format(maxv)
	thr = maxv/1000.0
	h,w = magimg.shape
	mthr = np.zeros(h*w)
	mthr = mthr.reshape((h,w))
	mthr.fill(thr)
	rcomp = magimg > mthr
	#rcomp = rcomp * 1.0
	score = np.sum(rcomp)/float(h*w)
	
	#plt.subplot(122),plt.imshow(magimg, cmap = 'gray')
	#plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
	fm = variance_of_laplacian(snapg)
	#print fm
	#plt.show()
	ratio = fm / score
	print "fft disp: {}  laplacian de {} e ratio {}".format(score,fm,ratio)
	text = "N"
 	#imagePath[-7:-4]
  	cv2.putText(image,str(conta) , (5, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
  	cv2.putText(gray, str(conta), (5, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
	# if the focus measure is less than the supplied threshold,
	# then the image should be considered "blurry"
	if fm < args["threshold"]:
		text = "B"
 		cv2.putText(image, "{};{:4.0f}".format(wori,fm), (5, 56),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
 	else:
 		cv2.putText(image, "{};{:4.0f}".format(wori,fm), (5, 56),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1) 
 	if score < args["threshold2"]:					
 	 	cv2.putText(image, "{:5.3f}".format(score), (5, 36),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1) 
 	else:
 	 	cv2.putText(image, "{:5.3f}".format(score), (5, 36),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1) 

	# show the image
	print imagePath,"{}: {:.2f}".format(text, fm)	
	[h,w,c] = image.shape
	largeimg[ypos:ypos+h,xpos:xpos+w]=image
	largeimgGray[ypos:ypos+h,xpos:xpos+w]=gray
	xpos += 96
	
	if xpos >= 1440:
		xpos = 0
		ypos += 96
	conta += 1
	if conta == 150:
		cv2.imshow("Image", largeimg)
		cv2.imshow("Image Gray", largeimgGray)	
		key = cv2.waitKey(0)
		xpos = 0
		ypos = 0
		conta = 0
		largeimg = np.zeros((960,1440,3), np.uint8)
		largeimgGray = np.zeros((960,1440), np.uint8)
		
cv2.imshow("Image", largeimg)
cv2.imshow("Image Gray", largeimgGray)
key = cv2.waitKey(0)
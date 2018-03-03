#!/usr/bin/python

# Autor: Helder Gualberto Andrade Rodrigues Junior
# Copyright: Universidade de Sao Paulo/Huawei
# Biblioteca para pre processamento de deteccao facial com OpenCV
# Data: 2017/11
from detection import Detection, CVDetection
from multiprocessing import Process, Pool, Queue
from datetime import datetime
from common import clock

import numpy as np
import cv2.cv as cv

import cv2
import time



class Camera:
	def __init__(self, cam_url, id):
		self.url = cam_url
		self.id = id
		self.last_detected = None

	def set_last_detected(self, name):
		self.last_detected = name
		print self.last_detected

	def get_last_detected(self):
		return self.last_detected

class OpencvCamera(Process):

	def __init__(self,args, url, id, cascade, sync_queue):
		Process.__init__(self)
		self.camera = Camera(url,id)
		self.cascade = cascade
		self.detected_queue = sync_queue
		self.run_flag = True
		self.args = args

	def novoEquad(self,x1,y1,x2,y2,wib,heb):
		#funcao para gerar regiao de recorte valor incial em 1/7 aumentado para 1/6
	    #x1,y1,x2,y2 sao as posicoes da face obtida pelo opencv
	    #wib e heb sao a largura e altura da imagem
	    #este algoritmo acrescenta uma pequena margem a imagem

	    fatorenq = 6

	    dx = (x2-x1)/fatorenq
	    x1 = (x1 - dx) if (x1 - dx)>0 else 0
	    x2 = (x2 + dx) if (x2 + dx)<wib else wib
	    dy = (y2 -y1)/fatorenq
	    y1 = (y1 - dy) if (y1 - dy)>0 else 0
	    y2 = (y2 + 2*dy) if (y2 + 2*dy) < heb else heb
	    return x1,y1,x2,y2

	def adjust_gamma(self, image, gamma=1.0):
		# build a lookup table mapping the pixel values [0, 255] to
		# their adjusted gamma values
		invGamma = 1.0 / gamma
		table = np.array([((i / 255.0) ** invGamma) * 255
			for i in np.arange(0, 256)]).astype("uint8")
	 
		# apply gamma correction using the lookup table
		return cv2.LUT(image, table)

	def log_gamma_ajust(self, image):
		# build a lookup table mapping the pixel values [0, 255] to
		# their adjusted gamma values

		table = np.array([((i / 255.0) ** (2.1038037/np.log10(i+2))) * 255
			for i in np.arange(0, 256)]).astype("uint8")
	 
		# apply gamma correction using the lookup table
		return cv2.LUT(image, table)

	def detect(self, img, cascade):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#gray = cv2.equalizeHist(gray)
		rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, #2
										 minSize=(40, 40),  #(120, 120)
										 flags = cv.CV_HAAR_SCALE_IMAGE)
		if len(rects) == 0:
			return [],0.0
		rects[:,2:] += rects[:,:2]
		bdisp = 0.0
		if len(rects)>0:
			x1, y1, x2, y2 = rects[0]
			bdisp  =  cv2.Laplacian(gray[y1:y2,x1:x2],cv2.CV_64F).var()
		return rects,bdisp

	def rotate_img(self, mat, angle=270):
		rows,cols,c = mat.shape
		M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
		dst = cv2.warpAffine(mat,M,(cols,rows))
		return dst

	def correlation(self, mat, limit=50):

		histogram = cv2.calcHist([mat],[0],None,[256],[0,256])

		lighter = histogram[limit:]
		darker = histogram[0:limit]

		sum_light = np.sum(lighter)
		sum_dark = np.sum(darker)

		return sum_dark/sum_light	


	def end_process(self):
		self.run_flag = False


	def run(self):

		cap = cv2.VideoCapture(self.camera.url)

		if cap is None:
			print "Error opening the camera..."
			return

		#Faz uma primeira captura para obter o tamanho da imagem
		ret, frame = cap.read()

		#Gera um fator de reducao de tamanho de acordo com a imagem atual da camera
		fatorRed = 480.0/frame.shape[1]

		lastTimeDetect = 0.0 	#Mantem o tempo do ultimo individuo reconhecido
		frameCounter = 0		#Mantem o contador de frames para o delay por frames

		try:

			while self.run_flag:

				ret, frame = cap.read()
				if frame is None:
					print "Camera disconnected or video finished..."
					break

				if self.args.rotateIm:
					frame = self.rotate_img(frame)

				if self.args.gammaCorrection != 1.0:
					if self.correlation(frame) > 3.5:
						frame = self.adjust_gamma(frame, gamma=self.args.gammaCorrection) 
						#frame = self.log_gamma_ajust(frame)
				

				frameCounter += 1 
				if (clock() - lastTimeDetect) < self.args.timeDelay or frameCounter < (self.args.frameDelay+1):
					continue

				reduzido = cv2.resize(frame,(0,0),fx=fatorRed,fy=fatorRed)
		        #detecta faces no frame
				rects,bdisp=self.detect(reduzido, self.cascade)

				#Retorna o numero de linhas, colunas e canais da imagem
				[heb,wib,pb] = frame.shape

				imgs = []

				for rcar in rects:
					# #verifica a quanto tempo uma pessoa nao e detectada e se passar de 10s seta para None
					# if (clock() - lastTimeDetect) > 10:
					# 	lastPersonDetect = None

					#converte objeto cv::Rect para np array
					ncar = np.array(rcar)

					#Redimensiona o rect com o fatorRed
					ncar =  ncar / fatorRed

					#Obtem os pontos do rect individualmente
					[xi1, yi1, xi2, yi2] = ncar.astype(int)
					#Algoritmo que acrescenta uma margem ao rect do opencv
					x1,y1,x2,y2 = self.novoEquad(xi1,yi1,xi2,yi2,wib,heb)
					#efetua o crop da imagem com o novo tamanho
					vis_roi = frame[y1:y2, x1:x2]
					#Obtem o tamanho da nova imagem (altura, largura e canais)
					roih,roiw,roic = vis_roi.shape

					#Verifica se o tamanho da imagem esta dentro do range para processamento rapido
					#se nao, redimensiona a imagem ja cropada
					if roiw > 240:
						#gera o fator de compressao
						fator = 180.0/float(roiw)
						#Faz o redimensionamento da imagem cropada
						vis_roi = cv2.resize(vis_roi,(0,0),fx=fator,fy=fator)

					#print "Recorte para processamento em ({},{})({},{})".format(x1,y1,x2,y2)
					imgs.append(vis_roi.copy())

				if len(imgs) > 0:
					det = CVDetection(self.camera, imgs, frame.copy(), datetime.now())
					self.detected_queue.put(det)
					frameCounter = 0
					lastTimeDetect = clock()

			cap.release()

		except KeyboardInterrupt:
			self.run_flag = False
			pass
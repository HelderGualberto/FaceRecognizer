import cv2
import numpy as np
import argparse

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

cap = cv2.VideoCapture(args.video)

ret, frame = cap.read()

while True:
	ret, frame = cap.read()
	
	if frame is None:
		print "Camera disconnected or video finished..."
		break

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
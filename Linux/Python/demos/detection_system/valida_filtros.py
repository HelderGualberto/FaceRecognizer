from OpencvCamera import Camera
from multiprocessing import Queue
from RecogSystem import RecogSystem
from OpenSSL import SSL
from pymongo import MongoClient
from detection import CVDetection
from datetime import datetime

import cv2, time
import argparse
import websocket
import ssl
import os


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	
	parser.add_argument('--cascade', type=str,
					help="cascade haar detector",
					default='../../../data/haarcascades/haarcascade_frontalface_alt.xml')
	parser.add_argument('--rotateIm', type=bool, help="Rotacao da camera",
					default=False)

	parser.add_argument('--fatorRed', type=float , help="Reducao para otimizacao de processamento",
					default= 0.25)#0.25

	parser.add_argument('--imgLocation', type=str,help="Local das imagens para serem validadas",default="./data/input/") #


	args = parser.parse_args()
	
	sync_queue_in = Queue()
	sync_queue_out = Queue()

	img_names = os.listdir(args.imgLocation)

	
	c =0 

	for name in img_names:
		imgs = []
		path = args.imgLocation + name
		mat = cv2.imread(path)

		imgs.append(mat)
		cam = Camera("Validation",name)
		det = CVDetection(cam, imgs, mat, datetime.now())

		sync_queue_in.put(det)

	print img_names

	recog_sys = RecogSystem(sync_queue_in, sync_queue_out)
	recog_sys.start()

	rec = 0
	desc = 0
	fp = 0

	while True:
		try:
			time.sleep(0.01)
			if not sync_queue_out.empty():
				c+=1
				det = sync_queue_out.get()
				if det.name == 'helder':
					rec +=1
				elif det.name == 'Desconhecido':
					desc +=1
				

				det.save_prediction()

		except KeyboardInterrupt:
			recog_sys.end_process()
			break

	print "Reconhecido: " + str(rec)
	print "Desconhecido: " + str(desc)
	print "Falso positivo: "+ str(c-rec-desc)

	recog_sys.join()
		


	

from OpencvCamera import OpencvCamera
from multiprocessing import Queue
from RecogSystem import RecogSystem
from OpenSSL import SSL
from pymongo import MongoClient


import cv2, time
import argparse
import websocket
import ssl


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--video',
					help="url do video",
					action='append')
	parser.add_argument('--cascade', type=str,
					help="cascade haar detector",
					default='../../../data/haarcascades/haarcascade_frontalface_alt.xml')
	parser.add_argument('--rotateIm', type=bool, help="Rotacao da camera",
					default=False)
	parser.add_argument('--fatorRed', type=float , help="Reducao para otimizacao de processamento",
					default= 0.25)#0.25
	parser.add_argument('--kurento', type=str,help="IP do servidor kurento",default="172.16.0.2")

	parser.add_argument('--timeDelay',type=float,help="Efetua delay de processamento com o tempo especificado",default=0.0)

	parser.add_argument('--frameDelay',type=int,help="Efetua delay de processamento com o numero de frames especificados",default=0)

	parser.add_argument('--gammaCorrection',type=float,help="Valor de gamma para algoritmo de correcao de luminosidade",default=1.0)

	parser.add_argument('--id',
					help="identificador de geolocalizacao da camera",
					action='append')

	parser.add_argument('--useKurento',type=int,help="Flag para utilizar ou nao o kurento",default=1)

	parser.add_argument('--cuda', type=bool, default=True)


	args = parser.parse_args()

	database_conn = MongoClient("mongodb://172.16.0.3:27017")
	reffacedb = database_conn.reffacedb

	#Process list
	opencv_cam_list = []

	if (args.video == None or args.id == None) or len(args.video) != len(args.id):
		print "Entrada invalida!"
		print "Para cada camera indique um identificador --id string_cam_id "
		quit()


	#Configurando Kurento
	useKurento = args.useKurento
	register = False
	learn = True
	nome_pessoa = 'Anderson'

	if useKurento:
		kurento="wss://"+args.kurento+":8443/media"
		cert = ssl.get_server_certificate((args.kurento,8443), ssl_version=ssl.PROTOCOL_SSLv23)
		lf = (len(ssl.PEM_FOOTER)+1)
		if cert[0-lf] != '\n':
			cert = cert[:0-lf]+'\n'+cert[0-lf:]                
		ws = websocket.create_connection(kurento, sslopt={"cert_reqs": ssl.CERT_NONE})

	cascade = cv2.CascadeClassifier(args.cascade)

	sync_queue_in = Queue()
	sync_queue_out = Queue()


	for i in range(len(args.video)):
		p = OpencvCamera(args, args.video[i], args.id[i],cascade,sync_queue_in)
		p.start()
		opencv_cam_list.append(p)

	recog_sys = RecogSystem(sync_queue_in, sync_queue_out)
	recog_sys.start()

	count = 0
	while True:
		try:
			time.sleep(0.1)
			if not sync_queue_out.empty():
				det = sync_queue_out.get()
				det.save_prediction()
				if det.prob >= 98.0 and learn:
					det.send_detected_to_register(reffacedb.base)

				#Codigo para cadastrar um novo usuario atraves das cameras de vigilancia
				if register and det.name == 'Desconhecido':
					det.name = nome_pessoa
					det.send_detected_to_register(reffacedb.base)

				count +=1

				if det.alert and useKurento:
					if useKurento:
						det.send_alert_to_kurento(ws)
					if det.name != 'Desconhecido':
						det.save_detection_on_database(reffacedb.detections)

		except KeyboardInterrupt:
			for cam in opencv_cam_list:
				cam.end_process()
			recog_sys.end_process()
			print "Counter: " + str(count)
			break

	for cam in opencv_cam_list:
		cam.join()
		


	

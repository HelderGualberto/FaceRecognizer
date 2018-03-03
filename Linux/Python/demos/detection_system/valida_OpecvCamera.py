from OpencvCamera import OpencvCamera
from multiprocessing import Queue
from RecogSystem import RecogSystem
from OpenSSL import SSL


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

	parser.add_argument('--id',
					help="identificador de geolocalizacao da camera",
					action='append')

	args = parser.parse_args()


	#Configurando Kurento
	useKurento = True

	if useKurento:
		kurento="wss://"+args.kurento+":8443/media"
		cert = ssl.get_server_certificate((args.kurento,8443), ssl_version=ssl.PROTOCOL_SSLv23)
		lf = (len(ssl.PEM_FOOTER)+1)
		if cert[0-lf] != '\n':
			cert = cert[:0-lf]+'\n'+cert[0-lf:]                
		ws = websocket.create_connection(kurento, sslopt={"cert_reqs": ssl.CERT_NONE})


	url1 = "rtsp://admin:huawei123@10.1.1.0:554/LiveMedia/ch1/Media1"
	url2 = "rtsp://viwer:foscam123@10.1.1.5:554/videoMain"
	url3 = "rtsp://viwer:foscam123@10.1.1.4:554/videoMain"

	cpath = "../../../data/haarcascades/haarcascade_frontalface_alt.xml"
	cascade = cv2.CascadeClassifier(cpath)

	sync_queue_in = Queue()
	sync_queue_out = Queue()

	cam1 = OpencvCamera(args, url1, "cam1",cascade,sync_queue_in)
	cam2 = OpencvCamera(args, url2, "cam2",cascade,sync_queue_in)
	cam3 = OpencvCamera(args, url3, "cam3",cascade,sync_queue_in)

	recog_sys = RecogSystem(sync_queue_in, sync_queue_out)

	cam1.start()
	cam2.start()
	cam3.start()
	recog_sys.start()


	while True:
		
		try:
			time.sleep(0.01)
			if not sync_queue_out.empty():
				det = sync_queue_out.get()
				det.save_prediction()

				if det.alert:
					det.send_alert_to_kurento(ws)
					# if det.name != 'Desconhecido':
					# 	det.save_detection_on_database(reffacedb.detections)

		except KeyboardInterrupt:
			cam1.end_process()
			cam2.end_process()
			cam3.end_process()
			recog_sys.end_process()
			break


	cam1.join()
	cam2.join()
	cam3.join()


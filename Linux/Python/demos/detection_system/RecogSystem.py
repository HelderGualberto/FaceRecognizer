from multiprocessing import Process, Pool, Queue
from Recog import Classificador
from detection import Detection
from common import clock

class RecogSystem(Process):
	def __init__(self, sync_queue_in, sync_queue_out):
		Process.__init__(self)
		self.classif = Classificador()
		self.classif.loadClassif()
		self.sync_queue_in = sync_queue_in
		self.sync_queue_out = sync_queue_out
		self.run_flag = True

	def end_process(self):
		self.run_flag = False

	def recog_openface(self, detection):
		t = clock()
		dinfo = []
		for frame in detection.imgs:
			if self.classif.equadra(frame.copy()):
				

				#Classifica a imagem utilizando a rede neural do Openface
				#Chama a funcao no arquivo Recog.py
				#Retorna um rank de individuos possiveis, o primeiro tem maior probabilidade
				resp, dinfo  = self.classif.classifica()

				(nome,pb,reffile,tipoc) = resp[0]
				print "Encontrado {} em {} na camera {}".format(nome,detection.time, detection.camera.id)

				#Cria um objeto com as informacoes sobre a deteccao
				#Utilizado tambem para salvar a imagem e enviar para Kurento
				
				#Grava a imagem original da deteccao
				#d = Detection(detection.camera, nome, pb, detection.org_img, detection.time,send_alert)

				#Grava a imagem cropada da deteccao
				#d = Detection(detection.camera, nome, pb, frame.copy(), detection.time,True, dinfo)

				#Grava a imagem original da deteccao
				d = Detection(detection.camera, nome, pb, detection.org_img, detection.time,True, dinfo)

				self.sync_queue_out.put(d)

				dt = clock() - t
				print 'T Time',dt,' s'
				print "_____________________________________________________________________"

	def run(self):

		while self.run_flag:
		
			try:
				if not self.sync_queue_in.empty():
					det = self.sync_queue_in.get()
					self.recog_openface(det)
			except KeyboardInterrupt:
				break



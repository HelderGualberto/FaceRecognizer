from datetime import datetime
import cv2
import json
from time import clock
class Detection:

	def __init__(self, camera,name, prob,img,time,send_alert,dinfo):
		self.camera = camera
		self.name = name
		self.prob = prob*100
		self.img = img
		self.now = time
		self.alert = send_alert
		self.dinfo = dinfo #contem informacoes detelhadas sobre imagem detectada como representacao e angulos da face [rep, repF, angh, angv] angulo em graus
		self.arq = '{}.{:02d}.{:02d}.{:02d}.{:02d}_{}_pb{:02.0f}_{}.jpg'.format(
			clock(),#self.now.month,
			self.now.day,
			self.now.hour,
			self.now.minute,
			self.now.second,
			self.camera.id,
			self.prob,
			self.name)
	#salva a imagem de deteccao na pasta predic
	def save_prediction(self):
		arquivo='./predic/predic_' + self.arq
		cv2.imwrite(arquivo,self.img)

	def simplesVetor(self, rep):
	    res = []
	    for v in rep:
	        res.append(v)
	    return res

	#Registra imagem de pessoa detectada com grau de confianca acima de 98%. Isso funciona como um aprendizado 'burro' do individuo
	def send_detected_to_register(self, base):

		REP = 0; REPF = 1; ANGH = 2; ANGV = 3

		to_insert = {}
		to_insert['pessoa'] = self.name
		to_insert['arq'] = './register/' + self.arq
		to_insert['angH'] = self.dinfo[ANGH]
		to_insert['angV'] = self.dinfo[ANGV]
		to_insert['rep'] = self.simplesVetor(self.dinfo[REP])
		to_insert['repFlip'] = self.simplesVetor(self.dinfo[REPF])

		base.insert(to_insert)


	#Envia as informacoes de deteccao para o Kurento
	def send_alert_to_kurento(self, ws):
		payload = {'id': 'sendDetectionAlert', 'name':self.name, 'probability':self.prob, 'camera_id':self.camera.id, 'camera_url':self.camera.url, 'date':self.now.strftime('%D:%H:%M:%S')}
		data = json.dumps(payload)
		ws.send(data)

	#Envia as informacoes da deteccao para o mongodb
	def save_detection_on_database(self, collection):
		msg = {'name':self.name, 'prob':self.prob, 'camera_id':self.camera.id, 'camera_url':self.camera.url, 'date':self.now.strftime('%D:%H:%M:%S')}
		collection.insert_one(msg)

#Classe que detecmina a deteccao facial pelo Opencv para ser enviado ao DLIB
#Constitui de informacoes sobre a camera (id e URL), crop das faces detectadas e horarios da deteccao
class CVDetection(object):
	def __init__(self,camera,imgs,org_img,time):
		self.camera = camera
		self.imgs = imgs
		self.org_img = org_img
		self.time = time
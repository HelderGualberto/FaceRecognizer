#!/usr/bin/python

# Autor: Roberto Kenji Hiramatsu
# Copyright: Universidade de Sao Paulo/Huawei
# Servidor para processamento de requisicao de processamento de rede neural
# Data: 2016/11/23 - Inicial
#       2016/12/01 - Implementando tratador da rede deep
#       2016/12/02 - Implementando processamento para reconhecimento
#                    atraves de uma base
#       2017/03/06 - Arrumando caminho de litura de arquivo de dados
#       2017/03/29 - Padronizado para usar a versao Recog.py em util

import sys,os
import json
from libColetaFaceDB import MongoConn

import numpy as np

from twisted.internet import reactor
from twisted.python import log

from autobahn.twisted.websocket import WebSocketServerFactory, \
    WebSocketServerProtocol, \
    listenWS

import openface
from align_dlib import AlignDlib

from Recog import Classificador

import pymongo

import RepUtil

import cv2


basedir=os.path.join('..','..','data')

class RecogServerProtocol(WebSocketServerProtocol):
    def __init__(self):
        #conexao onde buscara imagem para ser classificada
        self.mdb     = None
        #classe usada para processamento e classificacao
        self.classif = None
        # alocado globalmente o extrador dlib
        self.align = AlignDlib(os.path.join(basedir,"models","dlib", "shape_predictor_68_face_landmarks.dat"))

        self.net = openface.TorchNeuralNet(os.path.join(basedir,"models", 'openface', #'treinado-jun16.t7'))
                                         'nn4.small2.v1.t7')
                              ,imgDim=96,
                              cuda=False)

    def onConnect(self, request):
        print("Client connecting: {0}".format(request.peer))

    def onMessage(self, payload, isBinary):
        raw = payload.decode('utf8')
        msg = json.loads(raw)
        print msg['type']
        if msg['type'] == "MONGOSRVINI":
            # inicializar o servidor mongodb onde sera baixado os arquivos
            resposta = self.initMongo(msg)
        elif msg['type'] == "MONGOPOSEDB":
            resposta = self.initMongoPose(msg)
        elif msg['type'] == "QUERYIMGREP":
            # processar uma imagem
            resposta = self.processImgRep(msg)
        elif msg['type'] == "QUERYIDREP":
            # processar as imagens de uma determinada identidade
            resposta = self.processIdRep(msg)
        elif msg['type'] == "QUERYIMGINFO":
            resposta = self.processImgInfo(msg)
        elif msg['type'] == "QUERYIDINFO":
            resposta = self.processIdInfo(msg)
        else:
            resposta = {'REASON':"UNKNOW parameters {}".format(msg.keys()),'RESPONSE':'ERROR'}
        resposta['type']=msg['type']
        self.sendMessage(json.dumps(resposta))

    def initMongo(self,msg):
        #resposta = resposta = {'RESPONSE':'ERROR','REASON':'MONGOSRVINI:Unknow parameters'+msg.keys()}
        if 'host' not in msg.keys() or 'port' not in msg.keys():
            resposta = {'RESPONSE':'ERROR','REASON':'Not have host or port field'}
        else:
            if 'base' not in msg.keys():
                msg['base']='facecoleta' # default database from libColeta
            print "mongodb://{}:{}".format(msg['host'],msg['port'])
            self.mdb = MongoConn(url="mongodb://{}:{}".format(msg['host'],msg['port']),
                                dbs=msg['base'])
            if self.mdb is not None:
                resposta = {'RESPONSE':'OK'}
            else:
                resposta = {'RESPONSE':'ERROR'}
        return resposta

    def initMongoPose(self,msg):
        #resposta = resposta = {'RESPONSE':'ERROR','REASON':'MONGOSRVINI:Unknow parameters'+msg.keys()}
        if 'host' not in msg.keys() or 'port' not in msg.keys():
            resposta = {'RESPONSE':'ERROR','REASON':'Not have host or port field'}
        else:
            texto = "Load face database classifier from mongodb://{}:{}".format(msg['host'],msg['port'])
            print texto
            try:
                self.classif=Classificador(self.align,self.net,host=msg['host'],port=msg['port'])
            except pymongo.errors.ServerSelectionTimeoutError:
                print "Can not connect to server mongodb://{}:{}".format(msg['host'],msg['port'])
                self.classif=None
            if self.classif is not None:
                resposta = {'RESPONSE':'OK','INFO':texto}
                try:
                    self.classif.loadClassif()
                except pymongo.errors.ServerSelectionTimeoutError:
                    texto= "Can not load data from server mongodb://{}:{}".format(msg['host'],msg['port'])
                    self.classif=None
                    resposta = {'RESPONSE':'ERROR','REASON':texto}
            else:
                resposta = {'RESPONSE':'ERROR'}
        return resposta

    def processImgRep(self,msg):
        #resposta = {'RESPONSE':'ERROR','REASON':'QUERYIMGREP:Unknow parameters:'+msg.keys()}
        if self.mdb is None:
            resposta = {'RESPONSE':'ERROR','REASON':'mongodb serve is not connected'}
        else:
            if 'filename' not in msg.keys():
                resposta = {'RESPONSE':'ERROR','REASON':'Not have filename field'}
            else:
                keys={'filename':msg['filename']}
                imgs=self.mdb.readImFromDB(keys=keys)
                if imgs is None:
                    resposta = {'RESPONSE':'ERROR','REASON':'Something wrong in mongodb'}
                elif len(imgs)<1:
                    resposta = {'RESPONSE':'ERROR','REASON':'Do not have file in db'}
                elif len(imgs) == 1:
                    for doc,img in imgs:
                        bb = self.align.getLargestFaceBoundingBox(img)
                        if bb is None:
                            print "O arquivo {} nao detectou face".format(doc["filename"])
                            resposta = {'RESPONSE':'ERROR','REASON':'Not found face'}
                            break
                        landmarks = self.align.findLandmarks(img, bb)
                        angcab,angvcab,pp = RepUtil.calcHVAngRosto(landmarks[0],landmarks[16],landmarks[27])
                        alignedFace = self.align.align(96, img, bb,
                                      landmarks=landmarks,
                                      landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                        rep  = self.net.forward(alignedFace)
                        resposta = {'RESPONSE':'OK','rep':rep.tolist(),
                                    'identity':doc['identity'],
                                    'angh':angcab,
                                    'angv':angvcab}
                else:
                    resposta = {'RESPONSE':'ERROR','REASON':'Multiples results docs'}
        return resposta

    def processImgInfo(self,msg):
        #resposta = {'RESPONSE':'ERROR','REASON':'QUERYIMGREP:Unknow parameters:'+msg.keys()}
        if self.mdb is None:
            resposta = {'RESPONSE':'ERROR','REASON':'mongodb serve is not connected'}
        else:
            if 'filename' not in msg.keys():
                resposta = {'RESPONSE':'ERROR','REASON':'Not have filename field'}
            else:
                keys={'filename':msg['filename']}
                imgs=self.mdb.readImFromDB(keys=keys)
                if imgs is None:
                    resposta = {'RESPONSE':'ERROR','REASON':'Something wrong in mongodb'}
                elif len(imgs)<1:
                    resposta = {'RESPONSE':'ERROR','REASON':'Do not have file in db'}
                elif len(imgs) == 1:
                    for doc,img in imgs:
                        bb = self.align.getLargestFaceBoundingBox(img)
                        if bb is None:
                            print "O arquivo {} nao detectou face".format(doc["filename"])
                            resposta = {'RESPONSE':'ERROR','REASON':'Not found face'}
                            break
                        landmarks = self.align.findLandmarks(img, bb)
                        angcab,angvcab,pp = RepUtil.calcHVAngRosto(landmarks[0],landmarks[16],landmarks[27])
                        alignedFace = self.align.align(96, img, bb,
                                      landmarks=landmarks,
                                      landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                        rep  = self.net.forward(alignedFace)
                        #repF = self.net.forward(cv2.flip(alignedFace))
                        # Somente sobre a imagem base se o espelhamento
                        # Reduz a metade a possibilidades
                        resp,nc = self.classif.buscaCandidato(rep, angcab, angvcab)

                        resposta = {'RESPONSE':'OK','rep':rep.tolist(),
                                    'identity':doc['identity'],
                                    'angh':angcab,
                                    'angv':angvcab}
                        countMat = 0
                        for nome,pb,reffile,tipoc,angs in resp:
                            if nome != "Desconhecido":
                                resposta["name{:02d}".format(countMat)]=nome
                                resposta["prob{:02d}".format(countMat)]=pb
                                resposta["tycl{:02d}".format(countMat)]=tipoc
                                anghe,angve = angs
                                resposta["angh{:02d}".format(countMat)]=anghe
                                resposta["angv{:02d}".format(countMat)]=angve
                                print "A imagem {} pode ser de {} com prob {:4.1f}".format(doc["filename"],nome,pb*100)
                                countMat += 1
                        resposta['nclassif']=countMat

                else:
                    resposta = {'RESPONSE':'ERROR','REASON':'Multiples results docs'}
        return resposta

    def processIdRep(self,msg):
        #resposta = {'RESPONSE':'ERROR','REASON':'QUERYIDREP:Unknow parameters:'+msg.keys()}
        if self.mdb is None:
            resposta = {'RESPONSE':'ERROR','REASON':'mongodb serve is not connected'}
        else:
            if 'identity' not in msg.keys():
                resposta = {'RESPONSE':'ERROR','REASON':'Not have filename field'}
            else:
                keys={'identity':msg['identity']}
                imgs=self.mdb.readImFromDB(keys=keys)
                if imgs is None:
                    resposta = {'RESPONSE':'ERROR','REASON':'Something wrong in mongodb'}
                elif len(imgs)<1:
                    resposta = {'RESPONSE':'ERROR','REASON':'Do not have file in db'}
                else:
                    resposta = {'RESPONSE':'OK'}
                    contap = 0
                    contaf = 0
                    seqf=[]
                    for doc,img in imgs:
                        bb = self.align.getLargestFaceBoundingBox(img)
                        if bb is None:
                            print "O arquivo {} nao detectou face".format(doc["filename"])
                            seqf.append(doc['seq'])
                            contaf += 1
                        else:
                            if "rep{:03d}".format(doc['seq']) in doc.keys():
                                resposta["rep{:03d}".format(doc['seq'])] = doc["rep{:03d}".format(doc['seq'])]
                                resposta["angh{:03d}".format(doc['seq'])] = doc["angh{:03d}".format(doc['seq'])]
                                resposta["angv{:03d}".format(doc['seq'])] = doc["angv{:03d}".format(doc['seq'])]
                            else:
                                landmarks = self.align.findLandmarks(img, bb)
                                angcab,angvcab,pp = RepUtil.calcHVAngRosto(landmarks[0],landmarks[16],landmarks[27])
                                alignedFace = self.align.align(96, img, bb,
                                      landmarks=landmarks,
                                      landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                                rep  = self.net.forward(alignedFace)
                                resposta["rep{:03d}".format(doc['seq'])] = rep.tolist()
                                resposta["angh{:03d}".format(doc['seq'])] = angcab
                                resposta["angv{:03d}".format(doc['seq'])] = angvcab
                                self.mdb.db.fs.files.update({'_id':doc['_id']},{ "$set":
                                                        {"rep{:03d}".format(doc['seq']):rep.tolist(),
                                                         "angh{:03d}".format(doc['seq']):angcab,
                                                         "angv{:03d}".format(doc['seq']):angvcab}})


                            contap += 1
                    if (contap+contaf)%10 == 0:
                            texto = "Processed {} images".format((contap+contaf))
                            print texto
                            parcial={'RESPONSE':'INPROGRESS',
                                      'INFO':texto}
                            self.sendMessage(json.dumps(parcial))
                    resposta['countp']=contap
                    resposta['seqf']=np.array(seqf).tolist()
                    resposta['countn']=contaf
                    resposta['identity']=doc['identity']
        return resposta

    def processIdInfo(self,msg):
        #resposta = {'RESPONSE':'ERROR','REASON':'QUERYIDREP:Unknow parameters:'+msg.keys()}
        if self.mdb is None:
            resposta = {'RESPONSE':'ERROR','REASON':'mongodb serve is not connected'}
        else:
            if 'identity' not in msg.keys():
                resposta = {'RESPONSE':'ERROR','REASON':'Not have filename field'}
            else:
                keys={'identity':msg['identity']}
                imgs=self.mdb.readImFromDB(keys=keys)
                if imgs is None:
                    resposta = {'RESPONSE':'ERROR','REASON':'Something wrong in mongodb'}
                elif len(imgs)<1:
                    resposta = {'RESPONSE':'ERROR','REASON':'Do not have file in db'}
                else:
                    resposta = {'RESPONSE':'OK'}
                    contap = 0
                    contaf = 0
                    seqf=[]
                    for doc,img in imgs:
                        bb = self.align.getLargestFaceBoundingBox(img)
                        if bb is None:
                            print "O arquivo {} nao detectou face".format(doc["filename"])
                            seqf.append(doc['seq'])
                            contaf += 1
                        else:
                            landmarks = self.align.findLandmarks(img, bb)
                            angcab,angvcab,pp = RepUtil.calcHVAngRosto(landmarks[0],landmarks[16],landmarks[27])
                            alignedFace = self.align.align(96, img, bb,
                                      landmarks=landmarks,
                                      landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                            rep  = self.net.forward(alignedFace)
                            resposta["rep{:03d}".format(doc['seq'])] = rep.tolist()
                            resposta["angh{:03d}".format(doc['seq'])] = angcab
                            resposta["angv{:03d}".format(doc['seq'])] = angvcab
                            resp,nc = self.classif.buscaCandidato(rep, angcab, angvcab)
                            countMat = 0
                            for nome,pb,reffile,tipoc,angs in resp:
                                if nome != "Desconhecido":
                                    resposta["name{:03d}.{:02d}".format(doc['seq'],countMat)]=nome
                                    resposta["prob{:03d}.{:02d}".format(doc['seq'],countMat)]=pb
                                    resposta["tycl{:03d}.{:02d}".format(doc['seq'],countMat)]=tipoc
                                    anghe,angve = angs
                                    resposta["angh{:02d}".format(countMat)]=anghe
                                    resposta["angv{:02d}".format(countMat)]=angve
                                    print "A imagem {} pode ser de {} com prob {:4.1f}".format(doc["filename"],nome,pb*100)
                                    countMat += 1
                            resposta["nclassif".format(doc['seq'])]=countMat
                            contap += 1
                    if (contap+contaf)%10 == 0:
                            texto = "Processed {} images".format((contap+contaf))
                            print texto
                            parcial={'RESPONSE':'INPROGRESS',
                                      'INFO':texto}
                            self.sendMessage(json.dumps(parcial))
                    resposta['countp']=contap
                    resposta['seqf']=np.array(seqf).tolist()
                    resposta['countn']=contaf
                    resposta['identity']=doc['identity']
        return resposta

    def onClose(self,wasClean,code,reason):
        print "Connection closed with reason:{} code:{} is clean:{}".format(reason,code,wasClean)

if __name__ == '__main__':

    log.startLogging(sys.stdout)

    headers = {'MyCustomServerHeader': 'Foobar'}

    factory = WebSocketServerFactory(u"ws://0.0.0.0:8081")
    factory.protocol = RecogServerProtocol
    listenWS(factory)
    reactor.run()

#!/usr/bin/python

# Autor: Roberto Kenji Hiramatsu
# Copyright: Universidade de Sao Paulo/Huawei
# Servidor para processamento de requisicao de processamento de rede neural
# Data: 2016/11/23 - Inicial
#       2016/12/01 - Implementando tratador da rede deep

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

import RepUtil

import pymongo
# alocado globalmente o extrador dlib
align = AlignDlib(os.path.join("models","dlib", "shape_predictor_68_face_landmarks.dat"))

net = openface.TorchNeuralNet(os.path.join("models", 'openface', #'treinado-jun16.t7'))
                                         'nn4.small2.v1.t7')
                              ,imgDim=96,
                              cuda=False)

class RecogServerProtocol(WebSocketServerProtocol):
    def __init__(self):
        self.mdb = None

    def onConnect(self, request):
        print("Client connecting: {0}".format(request.peer))

    def onMessage(self, payload, isBinary):
        raw = payload.decode('utf8')
        msg = json.loads(raw)
        print msg['type']
        if msg['type'] == "MONGOSRVINI":
            # inicializar o servidor mongodb onde sera baixado os arquivos
            resposta=self.initMongo(msg)

        elif msg['type'] == "QUERYIMGREP":
            # processar uma imagem
            resposta=self.processImgRep(msg)
        elif msg['type'] == "QUERYIDREP":
            # processar as imagens de uma determinada identidade
            resposta=self.processIdRep(msg)
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
            try:
                self.mdb = MongoConn(url="mongodb://{}:{}".format(msg['host'],msg['port']),
                                dbs=msg['base'])
            except:
                mdburi = "mongodb://{}:{}".format(msg['host'],msg['port'])
                print mdburi
                resposta = {'RESPONSE':'ERROR',
                            'REASON':"Can not connect to mongodb {}".format(mdburi)}
                self.mdb = None
                return resposta
            if self.mdb is not None:
                resposta = {'RESPONSE':'OK'}
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
                        bb = align.getLargestFaceBoundingBox(img)
                        if bb is None:
                            print "O arquivo {} nao detectou face".format(doc["filename"])
                            resposta = {'RESPONSE':'ERROR','REASON':'Not found face'}
                            break
                        landmarks = align.findLandmarks(img, bb)
                        angcab,angvcab,pp = RepUtil.calcHVAngRosto(landmarks[0],landmarks[16],landmarks[27])
                        alignedFace = align.align(96, img, bb,
                                      landmarks=landmarks,
                                      landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                        rep  = net.forward(alignedFace)
                        resposta = {'RESPONSE':'OK','rep':rep.tolist(),
                                    'identity':doc['identity'],
                                    'angh':angcab,
                                    'angv':angvcab}
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
                        bb = align.getLargestFaceBoundingBox(img)
                        if bb is None:
                            print "O arquivo {} nao detectou face".format(doc["filename"])
                            seqf.append(doc['seq'])
                            contaf += 1
                        else:
                            landmarks = align.findLandmarks(img, bb)
                            angcab,angvcab,pp = RepUtil.calcHVAngRosto(landmarks[0],landmarks[16],landmarks[27])
                            alignedFace = align.align(96, img, bb,
                                      landmarks=landmarks,
                                      landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                            rep  = net.forward(alignedFace)
                            resposta["rep{:03d}".format(doc['seq'])] = rep.tolist()
                            resposta["angh{:03d}".format(doc['seq'])] = angcab
                            resposta["angv{:03d}".format(doc['seq'])] = angvcab
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

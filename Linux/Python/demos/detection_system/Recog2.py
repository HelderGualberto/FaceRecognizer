#!/usr/bin/env python2
#
# Copyright 2015-2016 Escola Politecnica - Universidade de Sao Paulo
#
# Avaliacao de individual em relacao a base de referencia
# determina valores
# Data: 2016/08/23 - versao inicial
#       2016/10/13 - processamento usando regressao logistica

import os
import pickle
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))

#interface com base de referencia
# base tratada com rede neural nn4.smallv2
from libPoseDB import MongoConnOg
# base tratada com rede neural treinada v
from libPoseDB import MongoConnT1

import txaio
txaio.use_twisted()

#from autobahn.websocket import WebSocketServerProtocol, \
#    WebSocketServerFactory
#from twisted.python import log
#from twisted.internet import reactor

import argparse
import cv2
#import imagehash
#import json
#from PIL import Image
import numpy as np
import os
#import StringIO
#import urllib
#import base64

from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC

import matplotlib as mpl
mpl.use('Agg')
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm


import os.path
import openface

from datetime import datetime
mypath='./predic'
from os import listdir
from os.path import isfile, join
#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
import re

import RepUtil
from RepUtil import Face

#modelDir = os.path.join(fileDir, '..', '..', 'models')
#caminho para o modelo quando a arvore de diretorios pyde/demos estiver no mesmo nivel do openface
modelDir = os.path.join('..', '..', '..','data','models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, #'treinado-jun16.t7'))
                                         'nn4.small2.v1.t7'))
#'treinado-jun16.t7'
#parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
#                    default=os.path.join(openfaceModelDir, 'custo9.3.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--cuda', type=bool, default=False)
parser.add_argument('--unknown', type=bool, default=False,
                    help='Try to predict unknown people')
parser.add_argument('--port', type=int, default=9000,
                    help='WebSocket Port')

parser.add_argument('--video', type=str,
                    help="url do video",
                    #default='rtsp://admin:B30cd4Ro@192.168.10.180:554/LiveMedia/ch1/Media1')
                    default='rtsp://admin:B30cd4Ro@127.0.0.1:8554/LiveMedia/ch1/Media2')
parser.add_argument('--rotateIm', type=bool, help="Frame per Second on cam",
                    default=True)

parser.add_argument('--fatorRed', type=float , help="Reducao para otimizacao de processamento",
                    default= 0.50)


args = parser.parse_args()
# os argumentos estao indicado para local existente a localizacao
print("args:",args)

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                              cuda=args.cuda)


#carrega reg logit e pessoas com angulo maximo defaut de deteccao em 30 graus
#quando a maquina estiver em kurentofront
#mdbref = MongoConnOg(url="mongodb://mdb:27017")
#quando o local de processamento estiver no godzilla
#mdbref = MongoConnOg(url="mongodb://kurentofront.pad.lsi.usp.br:37027")
#mdbref = MongoConnT1(url="mongodb://kurentofront.pad.lsi.usp.br:37027")


mdbref = MongoConnOg(url="mongodb://192.168.10.236:27017")

class Faces:

    def __init__(self, identity, nome):
        self.reps = []
        self.nome = nome
        self.identity = identity

    def __repr__(self):
        return "{{id: {},nome:{}}}".format(
            str(self.identity),self.nome
        )

def agrupado(people,images):
    listap = []
    for idx , val in enumerate(people):
        listap.append(Faces(idx,people[idx]))
    for imgb in images.values():
        listap[imgb.identity].reps.append(imgb)
    return listap

class ImgDis:
    def __init__(self,distancia,arquivo,desvang):
        self.distancia = distancia
        self.arquivo = arquivo
        self.desvang = desvang
    def __repr__(self):
        return "{{a:{},d:{:4.2f},da:{}}}".format(
            self.arquivo,self.distancia,self.desvang
        )

# indice para rejeicao de procura antes 0.5
indrejec = 0.66



class Classificador:
    def __init__(self):
        self.align = openface.AlignDlib(args.dlibFacePredictor)
        self.net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                              cuda=args.cuda)
        self.loadClassif()

    def loadClassif(self):
        #pessoas registradas na base de referencia
        self.pessoasreg = mdbref.recupera()
        self.logreg     = mdbref.carregalogit()

        #=======================================================================
        # if os.path.isfile('people.pkl'):
        #     with open('people.pkl','rb') as inppeo:
        #         self.people=pickle.load(inppeo)
        #     del inppeo
        #     with open('images.pkl','rb') as inpimg:
        #         self.images=pickle.load(inpimg)
        #     del inpimg
        #     with open('svm.pkl','rb') as inpsvm:
        #         self.svm=pickle.load(inpsvm)
        #=======================================================================
        #    del inpsvm

    # para obter os pontos fiduciais e a face alinhada
    def equadra(self,imag):
        #=======================================================================
        # [he, wi,p] = imag.shape
        # rgbFrame = np.zeros((he, wi, 3), dtype=np.uint8)
        # rgbFrame[:, :, 0] = imag[:, :, 0]
        # rgbFrame[:, :, 1] = imag[:, :, 1]
        # rgbFrame[:, :, 2] = imag[:, :, 2]
        # bb = self.align.getLargestFaceBoundingBox(rgbFrame)
        #=======================================================================
        print "Recebido imagem de {} para processamento".format(imag.shape)
        bb = self.align.getLargestFaceBoundingBox(imag)
        bbs = [bb] if bb is not None else []

        for bb in bbs:
            print "Processando regiao encontrada de dlib {}".format(bb)
            self.landmarks = self.align.findLandmarks(imag, bb)
            self.alignedFace = self.align.align(args.imgDim, imag, bb,
                                      landmarks=self.landmarks,
                                      landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if self.alignedFace is None:
                continue
            return True
        return False

    def processaLogit(self,X):
        X=np.array(X).reshape(1,-1)
        probas = self.logreg.predict_proba(X)
        return probas

    #Seleciona as iimagens mais proximas relacionadas na base
    def buscaCandidato(self,rep,repF,angh,angv):
        res = [("Desconhecido",0.0,None,"?"),("Desconhecido",0.0,None,"?"),("Desconhecido",0.0,None,"?"),("Desconhecido",0.0,None,"?")]
        aprob = 0.0
        for nome in self.pessoasreg.keys():
            for d in self.pessoasreg[nome]:
                dangh = abs(angh-d["angH"])
                dangv = abs(angv-d["angV"])
                # se a distancia angular do rosto for maior que 30 na horizontal ou vertical entao nao processa
                if dangh > 30.0 or dangv > 30.0:
                    continue
                dist  = RepUtil.calcDistancia(rep,d["rep"])
                distF = RepUtil.calcDistancia(repF,d["repFlip"])
                X = [dist,dangh,dangv]
                mpb = self.processaLogit(X)
                pb = mpb[0,1]
                if aprob < pb and pb > indrejec:
                    aprob = pb
                    res[3] = res[2]
                    res[2] = res[1]
                    res[1] = res[0]
                    res[0] = (nome,pb,d["arq"],"NN")

                X = [distF,dangh,dangv]
                mpb = self.processaLogit(X)
                pb = mpb[0,1]
                if aprob < pb and pb > indrejec:
                    aprob = pb
                    res[3] = res[2]
                    res[2] = res[1]
                    res[1] = res[0]
                    res[0] = (nome,pb,d["arq"],"FF")

                dangh = abs(angh + d["angH"])
                dangv = abs(angv + d["angV"])
                if dangh > 30.0 or dangv > 30.0:
                    continue
                distR=RepUtil.calcDistancia(rep,d["repFlip"])
                distRF=RepUtil.calcDistancia(repF,d["rep"])

                X = [distR,dangh,dangv]
                mpb = self.processaLogit(X)
                pb = mpb[0,1]
                if aprob < pb and pb > indrejec:
                    aprob = pb
                    res[3] = res[2]
                    res[2] = res[1]
                    res[1] = res[0]
                    res[0] = (nome,pb,d["arq"],"NF")

                X = [distRF,dangh,dangv]
                mpb = self.processaLogit(X)
                pb = mpb[0,1]
                if aprob < pb and pb > indrejec:
                    aprob = pb
                    res[3] = res[2]
                    res[2] = res[1]
                    res[1] = res[0]
                    res[0] = (nome,pb,d["arq"],"FN")

        conta = 0
        (nomr,pr,ffn,ttn) = res[0]

        separa = []
        for (nom,pp,ff,tt) in res:
            if nomr == nom and nom != "Desconhecido":
                conta += 1
                separa.append((nom,pp,ff,tt))
        if conta == 0:
            return [("Desconhecido",0.0,None,"?")],0
        else:
            return separa,conta

    # carrega os arquivos relacionados
    def carregaImgRef(self,resp):
        for (nom,pp,ff,tt) in resp:
            mdbref.leArquivoST("ico."+ff)

    #apresenta a classificacao final da face
    def classifica(self):
        angcab,angvcab,pp = RepUtil.calcHVAngRosto(self.landmarks[0],self.landmarks[16],self.landmarks[27])
        if abs(angcab)>30 or abs(angvcab)>30:
            return [("Fora de 30 graus",0.0,None,"?")]

        rep  = self.net.forward(self.alignedFace)

        repF = self.net.forward(cv2.flip(self.alignedFace,1))

        resp,nc = self.buscaCandidato(rep,repF, angcab, angvcab)

        if nc > 0 :
            self.carregaImgRef(resp)

        return resp

        #=======================================================================
        # resultasvm = self.svm.predict(rep)
        # identity = resultasvm[0]
        # return self.people[identity]
        #=======================================================================


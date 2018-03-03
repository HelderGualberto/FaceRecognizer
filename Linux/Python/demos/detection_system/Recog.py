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
from common import clock, draw_str
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
parser.add_argument('--cuda', type=bool, default=True)

parser.add_argument('--port', type=int, default=9000,
                    help='WebSocket Port')

parser.add_argument('--video', type=str,
                    help="url do video",
                    #default='rtsp://admin:B30cd4Ro@192.168.10.180:554/LiveMedia/ch1/Media1')
                    default='rtsp://admin:B30cd4Ro@127.0.0.1:8554/LiveMedia/ch1/Media2')
parser.add_argument('--rotateIm', type=bool, help="Frame per Second on cam",
                    default=False)

parser.add_argument('--fatorRed', type=float , help="Reducao para otimizacao de processamento",
                    default= 0.5)

parser.add_argument('--kurento', type=str,help="IP do servidor kurento",default="172.16.0.2")

parser.add_argument('--timeDelay', type=float, help="Efetua delay de deteccao com o tempo especificado em segundos", default=0.0)

parser.add_argument('--frameDelay',type=int,help="Efetua delay de processamento com o numero de frames especificados",default=0)

parser.add_argument('--id',
                help="identificador de geolocalizacao da camera",
                action='append')

parser.add_argument('--gammaCorrection',type=float,help="Valor de gamma para algoritmo de correcao de luminosidade",default=1.0)

parser.add_argument('--useKurento',type=int,help="Flag para utilizar ou nao o kurento",default=1)


args = parser.parse_args()
# os argumentos estao indicado para local existente a localizacao
print("args:",args)

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                              cuda=args.cuda)

#Mantem conexao com o mongodb
mdbref = MongoConnOg(url="mongodb://172.16.0.3:27017")


indrejec = 0.66 # Indice de regeicao

#################################################################################################
class Faces:

    def __init__(self, identity, nome):
        self.reps = []
        self.nome = nome
        self.identity = identity

    def __repr__(self):
        return "{{id: {},nome:{}}}".format(
            str(self.identity),self.nome
        )
    #-----------------------------------********************---------------------------------------------------
def agrupado(people,images):
    listap = []
    for idx , val in enumerate(people):
        listap.append(Faces(idx,people[idx]))
    for imgb in images.values():
        listap[imgb.identity].reps.append(imgb)
    return listap
    #-----------------------------------********************---------------------------------------------------
class ImgDis:
    def __init__(self,distancia,arquivo,desvang):
        self.distancia = distancia
        self.arquivo = arquivo
        self.desvang = desvang
    def __repr__(self):
        return "{{a:{},d:{:4.2f},da:{}}}".format(
            self.arquivo,self.distancia,self.desvang
        )

#################################################################################################
class Classificador:

    def __init__(self):
        self.align = openface.AlignDlib(args.dlibFacePredictor)
        self.net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                              cuda=args.cuda)
        self.loadClassif()

    #-----------------------------------********************---------------------------------------------------
    def loadClassif(self):
        #pessoas registradas na base de referencia
        #Mantem dados dos cadastrados recuperados no mongodb.
        #um hash com a chave "Pessoa" e o contudo das representacoes

        #retorna um hash baseado noas angulos horizontal e vertical
        self.pessoasreg = mdbref.recuperaHash()

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
    #-----------------------------------********************---------------------------------------------------
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
        #print "Recebido imagem de {} para processamento".format(imag.shape)
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
    #-----------------------------------********************---------------------------------------------------
    def processaLogit(self,X):
        X=np.array(X).reshape(1,-1)
        probas = self.logreg.predict_proba(X)
        return probas
	
    #-----------------------------------********************---------------------------------------------------
    #Seleciona as iimagens mais proximas relacionadas na base
    def buscaCandidato(self,rep,repF,angh,angv):
        res = [("Desconhecido",0.0,None,"?"),("Desconhecido",0.0,None,"?"),("Desconhecido",0.0,None,"?"),("Desconhecido",0.0,None,"?")]
        aprob = 0.0
        #for nome in self.pessoasreg.keys():
        angh = angh*45.0
        angv = angv*45.0
        for d in self.pessoasreg[RepUtil.getHashByAngle(angh,15,45)][RepUtil.getHashByAngle(angv,20,40)]:
            nome = d["pessoa"]
            # print "angh: ", angh
            # print "d[\"angH\"]", d["angH"]

            # print "angv: ", angv
            # print "d[\"angV\"]", d["angV"]
            
            dangh = abs(angh-d["angH"])
            dangv = abs(angv-d["angV"])
            # se a distancia angular do rosto cadastrado em relacao ao rosto capturado for maior que 30 na horizontal ou vertical entao nao processa
            # if dangh > 30.0 or dangv > 30.0:
            #     continue

	#Calcula distancia da imagem obtida com a do banco de dados
            dist  = RepUtil.calcDistancia(rep,d["rep"])

	#Calcula distancia da imagem flipada obtida com a do banco de dados flipada
            distF = RepUtil.calcDistancia(repF,d["repFlip"])
           
	#vetor com dados de distancia, angulo vertical e angulo horizontal		
            X = [dist,dangh,dangv]

	#calcula a probabilidade para a imagem normal
            mpb = self.processaLogit(X)
            pb = mpb[0,1]
	
	#verifica se a probabilidade e maior que o indice de rejeicao                
            if aprob < pb and pb > indrejec:
                aprob = pb
                res[3] = res[2]
                res[2] = res[1]
                res[1] = res[0]
                res[0] = (nome,pb,d["arq"],"NN")

	#calcula a probabilidade para a imagem flipada
            X = [distF,dangh,dangv]
            mpb = self.processaLogit(X)
            pb = mpb[0,1]

	#verifica se a probabilidade e maior que o indice de rejeicao                
            if aprob < pb and pb > indrejec:
                aprob = pb
                res[3] = res[2]
                res[2] = res[1]
                res[1] = res[0]
                res[0] = (nome,pb,d["arq"],"FF")

            dangh = abs(angh + d["angH"])
            dangv = abs(angv + d["angV"])
            # if dangh > 30.0 or dangv > 30.0:
            #     continue

	#Calcula distancia da imagem obtida com a do banco de dados flipada
            distR=RepUtil.calcDistancia(rep,d["repFlip"])

	#Calcula distancia da imagem flipada obtida com a do banco de dados normal
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
	#Estrutura 
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

    #-----------------------------------********************---------------------------------------------------
    # carrega os arquivos relacionados
    def carregaImgRef(self,resp):
        for (nom,pp,ff,tt) in resp:
            mdbref.leArquivoST("ico."+ff)

    #-----------------------------------********************---------------------------------------------------
    #apresenta a classificacao final da face
    def classifica(self):
        angcab,angvcab,pp = RepUtil.calcHVAngRosto(self.landmarks[0],self.landmarks[16],self.landmarks[27])
        #print "Angulo Horizontal: "+str(45.0*angcab)
        #print "Angulo vertical: " + str(45.0*angvcab)

        # if abs(angcab)>30 or abs(angvcab)>30:
        #     return [("Fora de 30 graus",0.0,None,"?")]

        #Envia a imagem da face alinhada para a rede neural
        #O resultado da funcao eh um vetor de 128 posicoes com as caracteristicas da face
        rep  = self.net.forward(self.alignedFace)

        #Envia a imagem da face alinhada com flip horizontal para a rede neural
        #O resultado da funcao eh um vetor de 128 posicoes com as caracteristicas da face
        repF = self.net.forward(cv2.flip(self.alignedFace,1))

        #Utiliza o vetor de 128 posicoes para compara os individuos (utiliza regressao logistica)
        resp,nc = self.buscaCandidato(rep,repF, angcab, angvcab)
	     
        
        #Trecho de codigo para insercao das novas imagens em tempo real na lista de representacoes
        (nome,pb,reffile,tipoc) = resp[0]
        if pb > 98.0:
            to_insert = {}
            to_insert['pessoa'] = nome
            to_insert['angH'] = 45.0*angcab
            to_insert['angV'] = 45.0*angvcab
            to_insert['rep'] = rep
            to_insert['repFlip'] = repF
            self.pessoasreg[RepUtil.getHashByAngle(45.0*angcab,15,45)][RepUtil.getHashByAngle(45.0*angvcab,20,40)].append(to_insert)


        if nc > 0 :
            self.carregaImgRef(resp)

        return resp, [rep, repF, 45.0*angcab, 45.0*angvcab]

        #=======================================================================
        # resultasvm = self.svm.predict(rep)
        # identity = resultasvm[0]
        # return self.people[identity]
        #=======================================================================
#################################################################################################


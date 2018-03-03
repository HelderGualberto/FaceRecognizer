#!/usr/bin/python

# Autor: Roberto Kenji Hiramatsu
# Copyright: Universidade de Sao Paulo/Huawei
# Script para analisar comportamento da rede neural em relacao a qualidade de imagem e de pose
# Data: 2016/08/17
#       2017/03/16 correcoes para uso da nova estrutura do svn
#       2017/03/23 Versao simpleficada se inserir dados para analise de qualidade

import os,sys
import re

from pexpect.ANSI import DoEmit
from scipy.special._ufuncs import y1
#fileDir = os.path.dirname(os.path.realpath(__file__))
#sys.path.append(os.path.join(fileDir, "..", ".."))
fileDir = os.path.join("..","..","data")
import argparse
import openface

from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC

from imutils import paths
from os import listdir
from os.path import isfile, join
import cv2
import cv2.cv as cv
import numpy as np
import imagehash
from PIL import Image

import json

import RepUtil
from RepUtil import Face

import math
from scipy.optimize import fsolve


import thread

import libPoseDB
from libPoseDB import MongoConnOg


modelDir = os.path.join(fileDir,  'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir,
                                         #'treinado-jun16.t7'))
                                         'nn4.small2.v1.t7'))
# 'treinado-jun16.t7'
#parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
#                   default=os.path.join(openfaceModelDir, 'custo9.3.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--cuda', type=bool, default=False)
parser.add_argument('--unknown', type=bool, default=False,
                    help='Try to predict unknown people')

parser.add_argument('--npath', type=str,
                    help="Caminho de processamento das imagens",
                    default='/srv/localhighr')

parser.add_argument('--cascade', type=str,
                    help="cascade haar detector",
                    default=os.path.join(fileDir,'haarcascades','haarcascade_frontalface_alt.xml'))

parser.add_argument('--mongoURL', type=str,
                    help="URL da base mongoDB - default em mdb:27017 quando executado de kurentofront",
                    default='mongodb://mdb:27017')

args = parser.parse_args()
print args

mdb = MongoConnOg(url=args.mongoURL)

mdb.carregaListaProcessada()

mypath = args.npath
print 'Processando diretorio ',mypath

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                              cuda=args.cuda)






def extraiRostoPDLib(rgbFrame):
    bb = align.getLargestFaceBoundingBox(rgbFrame)
    bbs = [bb] if bb is not None else []
    for bb in bbs:
        bl = (bb.left(), bb.bottom())
        tr = (bb.right(), bb.top())
        altura = bl[1] - tr[1]
        largura = tr[0] - bl[0]

        landmarks = align.findLandmarks(rgbFrame, bb)
        alignedFace = align.align(args.imgDim, rgbFrame, bb,
            landmarks=landmarks,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        return bb,landmarks,alignedFace
    return bb,None,None

def simplesVetor(rep):
    res = []
    for v in rep:
        res.append(v)
    return res

def analisaP(pathb,nameb):
    arquivos = os.listdir(pathb+'/'+nameb)
    expressao = '.+\.jpg'
    parte = re.compile(expressao)
    pessoa = []
    for arq in arquivos:
        if arq in mdb.listaProcessada:
            print arq,' foi processada '
            continue
        info = {}
        info['arq'] = arq
        info['pessoa'] = nameb
        if parte.search(arq):
            print 'Processando',arq
            img = cv2.imread(pathb+'/'+nameb+'/'+arq)
            bb,landmarks,alignedFace = extraiRostoPDLib(img)
            if bb is None:
                print arq, 'nao pode ser processado '
                continue
            thread.start_new_thread(libPoseDB.gravaIm,(mdb,alignedFace.copy(),'ico.'+arq,nameb,False))
            # obtendo os angulos da cabeca na imagem
            angcab,angvcab,pp = RepUtil.calcHVAngRosto(landmarks[0],landmarks[16],landmarks[27])
            angcab  *= 45
            angvcab *= 45
            info['angH']=angcab
            info['angV']=angvcab

            #obtem a representacao gerada da rede neural
            rep = net.forward(alignedFace)
            info['rep'] = simplesVetor(rep)

            #obtendo a area da imagem para criar imagens com baixa qualidade
            [heb,wib,pb] = img.shape
            x1=bb.left()
            x2=bb.right()
            y1=bb.top()
            y2=bb.bottom()
            x1,y1,x2,y2 = RepUtil.novoEquad(x1,y1,x2,y2,wib,heb)
            vis_roi = img[y1:y2, x1:x2]
            [vheb,vwib,pb] = vis_roi.shape
            thread.start_new_thread(libPoseDB.gravaIm,(mdb,vis_roi.copy(),arq,nameb,False))
            # gerando imagen espelhada horizontal para comporacao
            # obtendo representacao da imagem horizontal
            alignedFaceFli = cv2.flip(alignedFace,1)
            repFlip = net.forward(alignedFaceFli)
            info['repFlip'] = simplesVetor(repFlip)

            distFlip=RepUtil.calcDistancia(rep,repFlip)
            info['distFlip']=distFlip

            print 'Ang h/v',angcab,angvcab,' Distancia ',distFlip

            #print info
            pessoa.append(info)
            mdb.dbinfo.base.insert(info)

    return pessoa

if __name__=="__main__":
    dirs = os.listdir(mypath)
    for dirn in dirs:
        print dirn
        analisaP(mypath,dirn)

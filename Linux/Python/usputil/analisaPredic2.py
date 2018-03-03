#!/usr/bin/env python2
# Autor: Roberto Kenji Hiramatsu
#
# Copyright 2016 Universidade de Sao Paulo
#
# Script para analisar e criar fator de inclinacao h e v e
# nivel de dispersao
# calculos para posicionamento do rosto
# Incial Data: 2016-08-26
# 2016-08-27 - Versao com greacao de representacao no arquivo json



import os
import sys
from pexpect.ANSI import DoEmit
from scipy.special._ufuncs import y1
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))
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

modelDir = os.path.join(fileDir, '..', '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'treinado-jun16.t7'))
# 'nn4.small2.v1.t7'))
# 'treinado-jun16.t7'
#parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
#                   default=os.path.join(openfaceModelDir, 'custo9.3.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--cuda', type=bool, default=False)
parser.add_argument('--unknown', type=bool, default=False,
                    help='Try to predict unknown people')
parser.add_argument('--port', type=int, default=9000,
                    help='WebSocket Port')

parser.add_argument('--npath', type=str,
                    help="Caminho de processamento das imagens",
                    default='/srv/openface/demos/web2/predic')

parser.add_argument('--cascade', type=str,
                    help="cascade haar detector",
                    default=os.path.join(fileDir,'haarcascades','haarcascade_frontalface_alt.xml'))

parser.add_argument('--jsondb', type=str,
                    help="Armazenamento num arquivo json",
                    default='predicv2.json')

parser.add_argument('--csvdb', type=str,
                    help="Armazenamento num arquivo csv",
                    default='predicv2.csv')


args = parser.parse_args()

mypath = args.npath
# diretorio de arquivo processado
mypathproc = mypath+'_proc'

if not os.path.exists(mypathproc):
    print 'Criando diretorio ',mypathproc
    os.mkdir(mypathproc)


#arquivo para armazenar no formato json para uso posterior
jsondb = open(args.jsondb,'a')
csvdb = open(args.csvdb,'a')

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                              cuda=args.cuda)


def detect(img, cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    # 1.25 (120,120) 1
    rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2,
                                     minSize=(120, 120),
                                     flags = cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return [],0.0
    rects[:,2:] += rects[:,:2]
    bdisp = 0.0
    if len(rects)>0:
        x1, y1, x2, y2 = rects[0]
        bdisp  =  cv2.Laplacian(gray[y1:y2,x1:x2],cv2.CV_64F).var()
    return rects,bdisp

def calcHAngRosto(dre,drd,refang=math.pi/6.0):
    R = drd / dre
    func  = lambda ang : R -((math.cos(refang)*math.cos(ang)-math.sin(refang)*math.sin(ang))/(math.cos(refang)*math.cos(ang)+math.sin(refang)*math.sin(ang)))
    ang_g = math.pi/4.0 if dre > drd else -math.pi/4.0
    ang_s = fsolve(func,ang_g)
    return ang_s

# calcular angulos da cabeca usando 3 pontos fiduciais + reserva da ponta do nariz
def calcHVAngRosto(pre,prd,prc,refang=math.pi/6.0):
    x1,y1 = pre
    x2,y2 = prd
    x3,y3 = prc
    #calcula angulo da reta p1 p2
    ah = (float(y2-y1)/float(x2-x1))
    #interseccao do eixo y
    bh = float(y1)- ah*float(x1)
    # para calcular as posicoes projetadas e distancia do centro as bordas
    if ah == 0.0:
        # considera que x1 e x2 estao no mesmo valor de y
        xp = x3
        yp = y1
        dre = float(x3 - x1)
        drd = float(x2 - x3)
        dpp = abs(y3 - yp)
    else:
        # calcula p3 projetado na reta p1 p2
        av = -1.0/ah
        bv = y3 - av * float(x3)
        xp = (bv - bh)/(ah - av)
        yp = ah * xp + bh
        dre = RepUtil.calcDisDP([xp,yp],pre)
        drd = RepUtil.calcDisDP([xp,yp],prd)
        dpp = RepUtil.calcDisDP([xp,yp],prc)
    # angulo de inclinacao horizonal
    angh = calcHAngRosto(dre,drd)
    # distancia normalizada horizontal
    ded = (dre+drd)/math.cos(angh)
    # distancia de plano de profundidade centro
    dpc = ded * math.sin(refang)/2.0
    #print "dpp:{} dpc:{}".format(dpp,dpc)
    # angulo de inclinacao vertical de rosto
    angv = math.asin(dpp/dpc)
    angv = angv if yp > y3 else -angv
    return (angh[0]*180.0/math.pi),(angv*180.0/math.pi),(int(xp),int(yp))

def alteraEstado(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        lista,img = param
        ix = int (x/160)
        iy = int (y/180)
        conta = iy*8 + ix
        if lista[conta]["isok"] == 1:
            lista[conta]["isok"] = 0
        else:
            lista[conta]["isok"] = 1

        lt = (ix*160+2,iy*180+2)
        rb = ((ix+1)*160-2,(iy+1)*180-2)
        if lista[conta]["isok"] == 1:
            cv2.rectangle(img, lt , rb , color=(0,255,0),
                              thickness=3)
        else:
            cv2.rectangle(img, lt , rb , color=(0,0, 255),
                              thickness=3)

        cv2.namedWindow("quadro")
        cv2.setMouseCallback("quadro",alteraEstado,param=(lista,img))
        cv2.imshow("quadro",img)

        print conta,'Arq:',lista[conta]["filename"],' Sel Status:',lista[conta]["isok"]


if __name__ == '__main__':
        cascade = cv2.CascadeClassifier(args.cascade)
        largeimg = np.zeros((800,1280,3),np.uint8)
        xpos = 0
        ypos = 0
        listaInfoJ = []
        conta = 0
        for pimagem in paths.list_images(mypath):
            print pimagem
            rgbIn = cv2.imread(pimagem)
            [heb,wib,pb] = rgbIn.shape
            rects,bdisp=detect(rgbIn, cascade)
            for x1, y1, x2, y2 in rects:
                x1,y1,x2,y2 = RepUtil.novoEquad(x1,y1,x2,y2,wib,heb)
                vis_roi = rgbIn[y1:y2, x1:x2]
                [he,wi,pro] = vis_roi.shape
                if wi > 640:
                    fato = 640.0/float(wi)
                    rgbFrame  = cv2.resize(vis_roi,(0,0),fx=fato,fy=fato)
                else:
                    rgbFrame = vis_roi.copy()
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
                    idxB = RepUtil.calcBlur(alignedFace)
                    conta = 0
                    reff = [0,8,16,27,30,33,36,45,48,51,54,57,62,66]
                    for idxp in reff:
                    #for p in landmarks:
                        cv2.circle(rgbFrame,center=landmarks[idxp],radius=3,
                        #cv2.circle(rgbFrame,center=p,radius=3,
                               color=(255,204,102),thickness=-1)
                    dre = RepUtil.calcDisDP(landmarks[27],landmarks[0])
                    drd = RepUtil.calcDisDP(landmarks[27],landmarks[16])
                    dred = RepUtil.calcDisDP(landmarks[0],landmarks[16])
                    doe = RepUtil.calcDisDP(landmarks[27],landmarks[36])
                    dod = RepUtil.calcDisDP(landmarks[27],landmarks[45])
                    rze = dre / doe
                    rzd = drd / dod
                    rzed = rze /rzd

                    angcab,angvcab,pp = RepUtil.calcHVAngRosto(landmarks[0],landmarks[16],landmarks[27])
                    partes = pimagem.replace(".jpg","").split('predic_')
                    partes2 = partes[1].split("_",1)
                    cv2.circle(rgbFrame,center=pp,radius=3,
                               color=(0,0,255),thickness=-1)
                    [he,wi,pro] = rgbFrame.shape
                    if wi > 160:
                        fato = 160.0/float(wi)
                        recorte  = cv2.resize(rgbFrame,(0,0),fx=fato,fy=fato)
                    else:
                        recorte = rgbFrame.copy()
                    cv2.putText(recorte,partes2[0],(5,150),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,0),1)
                    cv2.putText(recorte,partes2[1],(5,15),cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,(0,255,0),2)
                    [h,w,c] = recorte.shape
                    largeimg[ypos:ypos+h,xpos:xpos+w]=recorte

                    #cv2.imshow("Pontos",rgbFrame)
                    #key = cv2.waitKey(0)
                    classifica = 1
                    caminhos = pimagem.split("/")
                    rep = net.forward(alignedFace)

                    infoj = RepUtil.geraJSON(partes2[0],partes2[1],angcab,angvcab,idxB,caminhos[len(caminhos)-1],classifica,netrep=rep)
                    listaInfoJ.append(infoj)
                    conta += 1


                    xpos += 160
                    if xpos > 1120:
                        xpos = 0
                        ypos += 180
                    if ypos >540:
                        cv2.namedWindow("quadro")
                        cv2.setMouseCallback("quadro",alteraEstado,param=(listaInfoJ,largeimg))
                        cv2.imshow("quadro",largeimg)
                        largeimg = np.zeros((800,1280,3),np.uint8)
                        key = cv2.waitKey(0)
                        conta = 0
                        print key
                        for infoj in listaInfoJ:
                            RepUtil.gravaJSON(jsondb,infoj)
                            if os.path.exists(mypath+'/'+infoj["filename"]):
                                os.rename(mypath+'/'+infoj["filename"],mypathproc+'/'+infoj["filename"])
                            else:
                                print 'Arquivo copiado ?',(mypath+'/'+infoj["filename"])
                        listaInfoJ = []
                        ypos = 0
        if conta > 0:
            cv2.namedWindow("quadro")
            cv2.setMouseCallback("quadro",alteraEstado,param=(listaInfoJ,largeimg))
            cv2.imshow("quadro",largeimg)
            key = cv2.waitKey(0)
            for infoj in listaInfoJ:
                RepUtil.gravaJSON(jsondb,infoj)
                if os.path.exists(mypath+'/'+infoj["filename"]):
                    os.rename(mypath+'/'+infoj["filename"],mypathproc+'/'+infoj["filename"])
                else:
                    print 'Arquivo copiado ?',(mypath+'/'+infoj["filename"])

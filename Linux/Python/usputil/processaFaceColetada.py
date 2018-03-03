#!/usr/bin/python

# Autor: Roberto Kenji Hiramatsu
# Copyright: Universidade de Sao Paulo/Huawei
# Processameento para separacao de dados de face coletados na camera da Huawei
# Os tratamentos executados sao a eliminacao de imagens nao detectadas com o dlib
# Registro de
# Data: 2016/10/10
#       2016/10/17 - verificacao de area de crop em relacao a imagem
#       2016/10/24 - restringindo processamento a dados noa processados de recorte da camera
#

from libColetaFaceDB import MongoConn
from libPoseDB import MongoConnOg
from libPoseDB import MongoConnT1

import pymongo
import gridfs

import os
import math

mdb = MongoConn()

import os,sys
import re

from pexpect.ANSI import DoEmit
from scipy.special._ufuncs import y1
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))
import argparse
import openface

import numpy as np

#from sklearn.decomposition import PCA
#from sklearn.grid_search import GridSearchCV
#from sklearn.manifold import TSNE
#from sklearn.svm import SVC

#from imutils import paths
#from os import listdir
#from os.path import isfile, join
import cv2
import cv2.cv as cv
import numpy as np
#import imagehash
#from PIL import Image

#import json

import RepUtil
#from RepUtil import Face

#import math
from scipy.optimize import fsolve

import thread

import re

#limite angular horizontal
limiteaceita=30
#verificacao considerando 5 parametros
isp5 = False

#carrega reg logit e pessoas
mdbref = MongoConnOg(url="mongodb://kurentofront.pad.lsi.usp.br:37027")
#mdbref = MongoConnT1(url="mongodb://kurentofront.pad.lsi.usp.br:37027")
#pessoas registradas na base de referencia


modelDir = os.path.join(fileDir, '..', '..', 'openface','models')
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




args = parser.parse_args()

mypath = args.npath
print 'Processando diretorio ',mypath

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                              cuda=args.cuda)



pessoasreg = mdbref.recupera()
logreg     = mdbref.carregalogit(nomearq='logreg.pkl',anglim=limiteaceita,isp5=isp5)

def processaLogit(X):
    X=np.array(X).reshape(1,-1)
    probas = logreg.predict_proba(X)
    return probas

def buscaCandidato(rep,angh,angv):
    res = [("Desconhecido",0.0,None,0.0,0.0),("Desconhecido",0.0,None,0.0,0.0),("Desconhecido",0.0,None,0.0,0.0)]
    aprob = 0.0
    for nome in pessoasreg.keys():
        for d in pessoasreg[nome]:
            #print d.keys()
            if abs(d["angH"]) >limiteaceita or abs(d["angV"])>limiteaceita :
                continue
            dangh = abs(angh-d["angH"])
            dangv = abs(angv-d["angV"])
            # se a distancia angular do rosto for maior que 30 na horizontal ou vertical entao nao processa
            #if dangh > 30.0 or dangv > 30.0:
            #    continue
            dist=RepUtil.calcDistancia(rep,d["rep"])
            if isp5:
                X = [dist,dangh,abs(angh),abs(d["angH"]),abs(angv),abs(d["angV"]),dangv]
            else:
                X = [dist,dangh,dangv]
            mpb = processaLogit(X)
            pb = mpb[0,1]
            if pb < 0.5:
                continue
            if aprob < pb:
                aprob = pb
                res[2] = res[1]
                res[1] = res[0]
                res[0] = (nome,pb,mdbref.leArquivoST("ico."+d["arq"]),d["angH"],d["angV"]) #
            dangh = abs(angh + d["angH"])
            dangv = abs(angv + d["angV"])
            #if dangh > 30.0 or dangv > 30.0:
            #    continue
            distR=RepUtil.calcDistancia(rep,d["repFlip"])
            if isp5:
                X = [distR,dangh,abs(angh),abs(d["angH"]),abs(angv),abs(d["angV"]),dangv]
            else:
                X = [distR,dangh,dangv]
            mpb = processaLogit(X)
            pb = mpb[0,1]
            if pb < 0.5:
                continue
            if aprob < pb:
                aprob = pb
                res[2] = res[1]
                res[1] = res[0]
                res[0] = (nome,pb,mdbref.leArquivoST("ico."+d["arq"]),d["angH"],d["angV"]) #
    conta = 0
    (nomr,pr,img,ah,av) = res[0]
    for (nom,pp,img,ah,av) in res:
        if nomr == nom and nom != "Desconhecido":
            conta += 1
    return res,conta

contaf = 0.0
somaf  = 0.0
somaqf = 0.0
def extraiRostoPDLib(rgbFrame):
    global contaf
    global somaf
    global somaqf
    bb = align.getLargestFaceBoundingBox(rgbFrame)
    bbs = [bb] if bb is not None else []
    for bb in bbs:
        bl = (bb.left(), bb.bottom())
        tr = (bb.right(), bb.top())
        altura = bl[1] - tr[1]
        largura = tr[0] - bl[0]
        h,w,c=rgbFrame.shape
        fatorr = float(largura)/float(w)
        contaf += 1
        somaf  += fatorr
        somaqf += fatorr*fatorr
        print "L: {} A:{} em {} x {} e fator {}".format(largura,altura,h,w,fatorr)
        landmarks = align.findLandmarks(rgbFrame, bb)
        alignedFace = align.align(args.imgDim, rgbFrame, bb,
            landmarks=landmarks,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        return bb,landmarks,alignedFace
    return bb,None,None


def detect(img, cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3,
                                     minSize=(200, 200),
                                     flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return [],0.0
    rects[:,2:] += rects[:,:2]
    bdisp = 0.0
    if len(rects)>0:
        x1, y1, x2, y2 = rects[0]
        bdisp  =  cv2.Laplacian(gray[y1:y2,x1:x2],cv2.CV_64F).var()
    return rects,bdisp

def novoEquad(areaequa,wib,heb):
    x1,y1,x2,y2 = areaequa
    dx = (x2-x1)/6
    x1 = (x1 - dx) if (x1 - dx)>0 else 0
    x2 = (x2 + dx) if (x2 + dx)<wib else wib
    dy = (y2 -y1)/6
    y1 = (y1 - dy) if (y1 - dy)>0 else 0
    y2 = (y2 + 2*dy) if (y2 + 2*dy) < heb else heb
    return x1,y1,x2,y2

expressao = 'frame.+.jpg'
parte = re.compile(expressao)
cascade = cv2.CascadeClassifier(args.cascade)

def processaImagem(d):
    try:
        mdb.leArquivo(d)
    except gridfs.errors.NoFile:
        retcur = mdb.db.fs.files.remove({"filename":d["filename"]})
        print "Removidos ",retcur
        return
    chaves = d.keys()
    if "tratado" in chaves:
        if d["tratado"]:
            print "{} foi processado no processamento de coleta".format(d["filename"])
            return

    img = cv2.imread('img/'+d["filename"])
    if img is None:
        os.remove('img/'+d["filename"])
        print "{} nao contem imagem jpg".format(d["filename"])
        retcur = mdb.db.fs.files.remove({"filename":d["filename"]})
        print "Removidos ",retcur
        return
    #===========================================================================
    # # para recortar a face do frame
    # if parte.search(d["filename"]):
    #     print '{} nao e recorte de face'.format(d["filename"])
    #     rects,bdisp = detect(img,cascade)
    #     hf,wf,pf = img.shape
    #     # recortando a imagem da area de interese
    #     for  areainte in rects:
    #         x1,y1,x2,y2 = novoEquad(areainte,wf,hf)
    #         img = img[y1:y2,x1:x2].copy()
    #         novoarq = d["filename"].replace("frame", "face")
    #         cv2.imwrite(novoarq,img)
    #         mdb.fs.put(open(novoarq,'rb'),filename=novoarq,localidade=d["localidade"],mesdia=d["mesdia"],horaminuto=d["horaminuto"],frame=d["frame"],
    #                 framebase=d["framebase"],area=d["area"])
    #         retcur = mdb.db.fs.files.remove({"filename":d["filename"]})
    #         os.remove(novoarq)
    #         print "{} substituido por {}".format(d["filename"],novoarq)
    #         break
    #===========================================================================

    isDetec = False
    bb,landmarks,alignedFace=extraiRostoPDLib(img)
    if bb is None:
        print "{} nao contem face detectada em dlib ".format(d["filename"])
        retcur = mdb.db.fs.files.remove({"filename":d["filename"]})
        print "Removidos ",retcur
        h,w,c = img.shape
        cv2.putText(img, "removido", (0,h/2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0,255), 2)
    else:
        isDetec = True
        print "Processando {}".format(d["filename"])
        # para recortar a face do frame
        if parte.search(d["filename"]):
            print '{} nao e recorte de face'.format(d["filename"])
            hf,wf,pf = img.shape
            print "enquadrando para {} {} {} {}".format(bb.left(),bb.top(),bb.right(),bb.bottom())
            x1,y1,x2,y2 = novoEquad((bb.left(),bb.top(),bb.right(),bb.bottom()),wf,hf)
            img = img[y1:y2,x1:x2].copy()
            novoarq = d["filename"].replace("frame", "face")
            cv2.imwrite(novoarq,img)
            mdb.fs.put(open(novoarq,'rb'),filename=novoarq,localidade=d["localidade"],mesdia=d["mesdia"],horaminuto=d["horaminuto"],frame=d["frame"],
                    framebase=d["framebase"],area=d["area"],tratado=True)
            retcur = mdb.db.fs.files.remove({"filename":d["filename"]})
            os.remove(novoarq)
            print "{} substituido por {}".format(d["filename"],novoarq)
        else:
            mdb.db.fs.files.update({"filename":d["filename"]},    {
                                                                   "$set": {
                                                                            "tratado":True
                                                                            },
                                                                   "$currentDate": {"lastModified": True}
                                                                   })

    cv2.imshow("Recorte",img)
    largeimg = np.zeros((576,396,3),np.uint8)
    ypos = 0
    nc = 0
    if bb is not None:
        angcab,angvcab,pp = RepUtil.calcHVAngRosto(landmarks[0],landmarks[16],landmarks[27])
        angcab  *= 45
        angvcab *= 45
        bindx=RepUtil.calcBlur(alignedFace)
        print "angH: {} angV: {} bindx:{}".format(angcab,angvcab,bindx)
        if abs(angcab)<limiteaceita and abs(angvcab)<limiteaceita:
            rep = net.forward(alignedFace)
            resp,nc = buscaCandidato(rep,angcab,angvcab)
            for nome,prob,img,ah,av in resp:
                print nome,prob
                if img is not None:
                    [h,w,p] = img.shape
                    largeimg[ypos:ypos+h,0:w]=img
                    texto = "N Prob:{:5.1f}% H:{:2.0f} V:{:2.0f}".format(prob*100,ah,av)
                    cv2.putText(largeimg,texto,(w,ypos+h/2),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,0),1)
                    ypos += h
            repFlip = net.forward(cv2.flip(alignedFace,1))
            resp,ncp = buscaCandidato(repFlip,-angcab,angvcab)
            nc += ncp
            print "Flip version"
            for nome,prob,img,ah,av in resp:
                print nome,prob
                if img is not None:
                    [h,w,p] = img.shape
                    largeimg[ypos:ypos+h,0:w]=img
                    texto = "F Prob:{:5.1f}% H:{:2.0f} V:{:2.0f}".format(prob*100,ah,av)
                    cv2.putText(largeimg,texto,(w,ypos+h/2),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(255,255,0),1)
                    ypos += h
            #if nc<2:
            #    cv2.waitKey(0)
    cv2.imshow("Resultado",largeimg)
    #if nc > 0 and nc < 5:
    #if isDetec:
    #    print "Area:{} frame:{}".format(d["area"],d["frame"])
    #    cv2.waitKey(0)
        #print d.keys()
    #else:
    cv2.waitKey(10)

    os.remove('img/'+d["filename"])



if __name__ == '__main__':
    global contaf
    global somaf
    global somaqf
    contaf = 0.0
    somaf  = 0.0
    somaqf = 0.0
    #,"horaminuto": {"$gt": 2100}
    retcur = mdb.db.fs.files.find({'tratado':False})  #{'mesdia':926}
    try:
        for d in retcur:
            processaImagem(d)
        if contaf > 0 :
            mediaf  = somaf / contaf
            desviof = math.sqrt((somaqf-somaf*somaf/contaf)/contaf)
            print"O fator medio e {}  com desvio de {} em {} ocorrencias".format(mediaf,desviof,contaf)
    except pymongo.errors.OperationFailure:
        print "Falhou no id {}".format(d)

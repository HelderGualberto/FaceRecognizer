#!/usr/bin/env python2
# Autor: Roberto Kenji Hiramatsu
#
# Copyright 2016 Universidade de Sao Paulo
#
# Extrator de face e informacoes relacionadas a qualidade da imagem e posicao 
# Data inicial: 2017-01-

# import the necessary packages
from imutils import paths
import argparse
import cv2
import numpy as np
import os
from os.path import join
from align_dlib import AlignDlib
from pymongo import MongoClient
from quality import variance_of_laplacian,fftScore
from pose    import calcHVAngRosto,solvePnPHPAng
from Recog   import Classificador

import openface

align = AlignDlib(join("..","..","data","models","dlib","shape_predictor_68_face_landmarks.dat"))
cascadep = join("..","..","data","haarcascades","haarcascade_frontalface_alt.xml")
cascade = cv2.CascadeClassifier(cascadep)

netp = join("..","..","data","models", 'openface', #'treinado-jun16.t7')
                                                   'nn4.small2.v1.t7')
# carrega a rede neural 
net = openface.TorchNeuralNet(netp)
# carrega o cassificador para faces baseada na nuvem usp 
classif=Classificador(align,net)
classif.loadClassif()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", 
            #default="D:\\Safety_City\\imagens\\qualityimg",
            default="/vmstor/openface/SafetyCity/videos/sequence/p.1222084100",
    help="path to input directory of images")
ap.add_argument("-o", "--output", 
default="/vmstor/openface/SafetyCity/imagens/",
    help="Tamanho da amostra maxima")
ap.add_argument("-u", "--mdburl",
default="mongodb://192.168.10.236:37027",
    help="URL de conecao ao banco de dados")
ap.add_argument("-s", "--size", type=int, default=240,
    help="Tamanho da amostra maxima")
ap.add_argument("--enabledb", type=bool, default=True)
args = vars(ap.parse_args())
print args
partc = args["images"].split('/')
princ=partc[len(partc)-1]
princ=princ.replace('.','_')
if args["enabledb"]:
    client = MongoClient(args["mdburl"])
    db = client["corredor"]
    colecao = db[princ]
    colecaoCan = db["{}_cand".format(princ)]

def detect(img, cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2,
                                     minSize=(90, 90))
    #print rects
    if len(rects) > 0:
        rects[:,2:] += rects[:,:2]
    print rects
    return rects

#gerar area de enquadramento
def novoEquadR(rect,wib,heb,fatorenq=6):
    x1,y1,x2,y2 = rect
    dx = (x2-x1)/fatorenq
    x1 = (x1 - dx) if (x1 - dx)>0 else 0
    x2 = (x2 + dx) if (x2 + dx)<wib else wib
    dy = (y2 -y1)/fatorenq
    y1 = (y1 - dy) if (y1 - dy)>0 else 0
    y2 = (y2 + 2*dy) if (y2 + 2*dy) < heb else heb
    return x1,y1,x2,y2


# Extracao de informacoes realcionados a face para anotacao no banco de dados
# @return - isok,img,rectimg,alarg,lapIdx,fftIdx,rep,(anghP,angvP,anghS,angvS)
def extratctFaceAli(sample):
    bb = align.getLargestFaceBoundingBox(sample)
    #processa para separar somente rosto mesmo extraido de haar
    if bb is None:
        return False,None,(0,0,0,0),0,0,0,None,(0,0,0,0)
    else:
        #obtendo as marcacoes da face
        landmarks = align.findLandmarks(sample, bb)
        #obtendo input para rede neural
        alignedFace = align.align(96,sample, bb,
                                      landmarks=landmarks,
                                      landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        #obtendo representacao
        rep  = net.forward(alignedFace)
        #obtendo a posicao do rosto na imagem
        anghP,angvP = calcHVAngRosto(landmarks)
        anghS,angvS = solvePnPHPAng(landmarks,sample.shape)

        alarg = (bb.right()-bb.left())
        gray  = cv2.cvtColor(sample[bb.top():bb.bottom(),bb.left():bb.right()].copy(), cv2.COLOR_BGR2GRAY)
        if gray is None:
            print "Something is wrong top:{} bottom:{} left:{} right:{} ".format(bb.top(),bb.bottom(),bb.left(),bb.right())
            return False,None,(0,0,0,0),0,0,0,None,(0,0,0,0)
        lapIdx = variance_of_laplacian(gray)
        fftIdx = fftScore(gray)
        if alarg > args["size"]:
            print "alarg:{}".format(alarg)
            fator = float(args["size"])/float(alarg)
            sample = cv2.resize(sample,(0,0),fx=fator,fy=fator)
            x1 = int (bb.left()*fator)
            x2 = int (bb.right()*fator)
            y1= int (bb.top()*fator)
            y2= int (bb.bottom()*fator)
        else:
            x1 = int (bb.left())
            x2 = int (bb.right())
            y1= int (bb.top())
            y2= int (bb.bottom())
        h,w,c = sample.shape
        x1,y1,x2,y2 = novoEquadR((x1,y1,x2,y2),w,h)    
        return True,sample[y1:y2,x1:x2].copy(),(x1,y1,x2,y2),alarg,lapIdx,fftIdx,rep.tolist(),(anghP,angvP,anghS,angvS)
    

#verify if output exits
if not os.path.exists(args["output"]):
   os.mkdir(args["output"])
else:
   if not os.path.isdir(args["output"]):
      print "Nao e diretorio a saida {} removendo e recriando como diretorio".format(args["output"])
      os.remove(args["output"])
      os.mkdir(args["output"])

# loop over the input image
conta = 0 
rotateIm=True
ordenado = sorted(paths.list_images(args["images"]))
for imagePath in ordenado:
    # load the image, convert it to grayscale, and compute the
    # focus measure of the image using the Variance of Laplacian
    # method
    #print imagePath
    image = cv2.imread(imagePath)
    if rotateIm:
        # teste com frame rotacionado
        (h, w) = image.shape[:2]
        Mtr = np.float32([[0,1,0],[-1,0,w]])
        image = cv2.warpAffine(image, Mtr, (h,w))    
    rects = detect(image,cascade)
    h,w,c = image.shape
    infoc={}
    parts=imagePath.split("/")
    infoc["fileIn"]=parts[len(parts)-1]
    infoc["rects"]= {}
    for idxr,rectCan in enumerate(rects):
        x1,y1,x2,y2 = novoEquadR(rectCan,w,h)
        infoc["rects"]["r{}".format(idxr)]=(x1,y1,x2,y2)
        sampCan = image[y1:y2,x1:x2].copy()
        #cv2.imshow("Sample",sampCan)
        #cv2.imshow("Thumb",cv2.resize(image,(0,0),fx=0.125,fy=0.125))
        #k=cv2.waitKey(0)
        ret,sample,rect,dlibWid,lapIdx,fftIdx,rep,angulos = extratctFaceAli(sampCan)
        if not ret:
            continue
    #cv2.imshow("Sample",image)
    #cv2.waitKey(0)
        arquivo = "{}-sample-{:05d}.jpg".format(princ,conta)
        cv2.imwrite(join(args["output"],arquivo),sample)
        info={}
        parts=imagePath.split("/")
        info["fileIn"]=parts[len(parts)-1]
        info["fileOut"]=arquivo
        xn1,yn1,xn2,yn2 =rect
        rect = (x1+xn1,y1+yn1,x1+xn2,y1+yn2)
        info["rectPos"]=rect
        info["dWidth"]=dlibWid
        info["lapIdx"]=lapIdx
        info["fftIdx"]=fftIdx
        anghP,angvP,anghS,angvS = angulos
        info["angHP"] = anghP
        info["angVP"] = angvP
        info["angHS"] = anghS
        info["angVS"] = angvS
        #resultado do processamento com rede neural referencia do openface
        info["rep_nn4_s2"]=rep
        #busca por cadidatos
        resp,nc = classif.buscaCandidato(rep, anghS, angvS)
        possi = 0
        for nome,pb,reffile,tipoc,angs in resp:
           if nome != "Desconhecido":
              info["nome_{}".format(possi)]=nome
              info["prob_{}".format(possi)]=pb
              info["ref_file_{}".format(possi)]=reffile
              info["tipo_{}".format(possi)]=tipoc
              info["ref_angs_{}".format(possi)]=angs
              possi += 1

        if args["enabledb"]:
            colecao.insert_one(info)
        print info
        conta += 1
    if args["enabledb"] and len(rects)>0:
        colecaoCan.insert_one(infoc)

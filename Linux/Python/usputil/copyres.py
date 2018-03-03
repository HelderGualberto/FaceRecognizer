#!/usr/bin/python

# Autor: Roberto Kenji Hiramatsu
# Copyright: Universidade de Sao Paulo/Huawei
# Script para criar um conjunto padronizado de resolucao adequada para imagem
# Data: 2017/03/22

import os
import cv2

basedir=os.path.join('..','..','data')
basedir=os.path.join('safetycity','Code','data')
cascadep=os.path.join(basedir,'haarcascades','haarcascade_frontalface_alt2.xml')
cascade = cv2.CascadeClassifier(cascadep)
#detecao da face com haar cascade em subretangulos da imagem
def detect(img,cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, #scaleFactor=1.05 2
                                     minSize=(100, 100))
    if len(rects)>1:
       x0,y0,w0,h0=rects[0]
       lr=w0
       for x,y,w,h in rects:
          if lr>w:
             x0,y0,w0,h0=x,y,w,h
    elif len(rects)>0:
       x0,y0,w0,h0=rects[0]
    else:
       return cv2.resize(img,(0,0),fx=0.5,fy=0.5)
       #h0,w0,c=img.shape
       #x0,y0=(0,0)
    imgr=img[y0-h0/3:y0+4*h0/3,x0-w0/3:x0+4*w0/3].copy()
    print((x0,y0,w0,h0))
    fator=200.0/float(w0)
    try:
       return cv2.resize(imgr,(0,0),fx=fator,fy=fator)
    except cv2.error:
       return cv2.resize(imgr,(0,0),fx=0.5,fy=0.5)


origem = 'localhighr'
destin = 'saida'

if not os.path.exists(destin):
   print destin
   os.mkdir(destin)

listaorig = os.listdir(origem)

for nome in listaorig:
   baseorig = os.path.join(origem,nome)
   listorig = os.listdir(baseorig) 
   basedest = os.path.join(destin,nome)
   if not os.path.exists(basedest):
      print basedest
      os.mkdir(basedest)
   for arq in listorig:
       arqorig=os.path.join(baseorig,arq)
       arqdest=os.path.join(basedest,arq)
       print "{}->{}".format(arqorig,arqdest)
       imgorig = cv2.imread(arqorig)
       print imgorig.shape       
       try:
          imgdest = detect(imgorig,cascade)
          cv2.imwrite(arqdest,imgdest)
       except cv2.error:
          print "{} erro ao processar".format(arqorig)



#!/usr/bin/env python2
# Autor: Roberto Kenji Hiramatsu
#
# Copyright 2016 Universidade de Sao Paulo
#
# Modelo para rastrear face para uso de classificacao sobre a camera
#
# Data inicial: 2016-11-09
#             
import os
import math
import argparse

from align_dlib import AlignDlib 

from common import clock
import cv2

#import RepUtil

import numpy as np
import math
from fractions import Fraction

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str,
                    help="Video a capturar",
                    #default='rtsp://kenji:6qi7j94i@192.168.10.181:554/H264') 
                    #default='rtsp://admin:B30cd4Ro@192.168.10.180:554/LiveMedia/ch1/Media1') 
                    #default='/home/yakumo/HBPVR')
                    default=0) #D:\\temp

#alt2 apresenta melhor desempenho para detecao de face
parser.add_argument('--cascade', type=str,
                    help="cascade haar detector",
                    #default='haarcascades/haarcascade_frontalface_alt_tree.xml')
                    #default='haarcascades/haarcascade_frontalface_alt2.xml')
                    default='haarcascades\\haarcascade_frontalface_alt2.xml') #D:\\app\\opencv31\\sources\\data\\
                    #default='haarcascades/haarcascade_frontalface_default.xml')
                    #default='haarcascades/haarcascade_frontalface_alt.xml')
                    #default='D:\\app\\opencv31\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml')


def detect(img, cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    #cv2.imshow("Cinza",gray)
    rects = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, #2
                                     minSize=(120, 120))
    #if len(rects) == 0:
    #    return []
    #print rects
    #rects[:,2:] += rects[:,:2]
    #print rects
    # preciso da imagem em escala de cinza para o rastreamento
    return rects,gray

def draw_rects(img, rects, color,listaDeRotulos=None):
    for x1, y1, w, h in rects:
        cv2.rectangle(img, (x1, y1), (x1+w, y1+h), color, 2)
        if listaDeRotulos is not None:
            cv2.putText(anotado,'R:{}'.format(listaDeRotulos[(x1,y1,w,h)]),(x1,(y1+20)),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(255,255,255),2)


def novoEquadR(rect,wib,heb,fatorenq=6):
    x1,y1,x2,y2 = rect
    dx = (x2-x1)/fatorenq
    x1 = (x1 - dx) if (x1 - dx)>0 else 0
    x2 = (x2 + dx) if (x2 + dx)<wib else wib
    dy = (y2 -y1)/fatorenq
    y1 = (y1 - dy) if (y1 - dy)>0 else 0
    y2 = (y2 + 2*dy) if (y2 + 2*dy) < heb else heb
    return x1,y1,x2,y2

#calcula distancia de dois pontos
def calcDistPoints(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    dx = float(x2-x1)
    dy = float (y2 - y1)
    return math.sqrt(dx*dx+dy*dy)

if __name__=="__main__":
    args = parser.parse_args()
    print "Carregando video em {}".format(args.video)
    cascade = cv2.CascadeClassifier(args.cascade)

    # objeto para controle de rastreamento de faces
    cap = cv2.VideoCapture(args.video)
    temframe=True
    conta = 0
    
    align = AlignDlib("shape_predictor_68_face_landmarks.dat")
    
    soma   = 0
    contad = 0
    while(temframe):
            t = clock()
            temframe, frame = cap.read()
            if temframe:
                conta += 1

                anotado=frame.copy()
                print "Frame {}".format(conta)
                rects,grayimg =detect(frame, cascade)
                
                hf,wf,c = frame.shape
                for x1,y1,w,h in rects:
                    x1,y1,x2,y2 = novoEquadR((x1,y1,x1+w,y1+h),wf,hf) 
                    roiframe = frame[y1:y2,x1:x2].copy()
                    w = abs(x2 - x1)
                    
                    fator = 1.0 if w < 180 else 180.0/float(w)
                    print "Fator: {}".format(fator)
                    #fator = 0.5
                    reduzido = cv2.resize(roiframe,(0,0),fx=fator,fy=fator)
                    #===============================================================
                    bb = align.getLargestFaceBoundingBox(reduzido)
                    if bb is not None:
                        xd1 = bb.left()
                        yd1 = bb.top()
                        xd2 = bb.right()
                        yd2 = bb.bottom()
                        wb = xd2 - xd1
                        xd1,yd1,xd2,yd2 = (np.array((xd1,yd1,xd2,yd2))/fator).astype(int) 
                        
                        draw_rects(anotado,[(x1+xd1,y1+yd1,xd2-xd1,yd2-yd1)],(0,255,255))
                #     print "Largest B:{}".format(bb)
                        landmarks = align.findLandmarks(reduzido, bb)
                        distl = calcDistPoints(landmarks[0],landmarks[16])
                        #w = x2 - x1
                        frac = distl/(w*fator)
                        soma   += frac
                        contad += 1
                        print "Valor de distl:{} , w: {} wb: {} e distl/w: {} media: {}".format(distl,w,wb,frac,soma/contad)
                        
                        alignedFace = align.align(96, reduzido, bb,
                                      landmarks=landmarks,
                                      landmarkIndices=align.OUTER_EYES_AND_NOSE)
                        cv2.imshow("Extraido",alignedFace)
                        
                #     #print "Landmarks: {}".format(landmarks)
                #     reff = [0,8,16,27,30,33,36,45,48,51,54,57,62,66]
                #     for idxp in landmarks:                    
                #         cv2.circle(anotado,center=idxp,radius=3,
                #                color=(255,204,102),thickness=-1)
                #===============================================================
                        
                #if len(rects)>0:
                #    draw_rects(anotado,rects,(0,0,255))
                cv2.imshow("Anotado",anotado)
                
                dt = clock() - t
                print "Total {} ms".format(dt*1000)
                # apresentado amostragem
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    # When everything done, release the capture
    cap.release()

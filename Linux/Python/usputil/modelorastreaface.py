#!/usr/bin/env python2
# Autor: Roberto Kenji Hiramatsu
#
# Copyright 2016 Universidade de Sao Paulo
#
# Modelo para rastrear face para uso de classificacao
#
# Data inicial: 2016-11-03
import os
import math
import argparse
from common import clock

import cv2
#import cv2.cv as cv

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str,
                    help="diretorio contendo video",
                    default='/home/yakumo/HBPVR')
                    #default="D:\\temp\\HBPVR")

parser.add_argument('--cascade', type=str,
                    help="cascade haar detector",
                    default='haarcascades/haarcascade_frontalface_alt.xml')
                    #default='D:\\app\\opencv31\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml')


def detect(img, cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    #cv2.imshow("Cinza",gray)
    rects = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2,
                                     minSize=(120, 120))
    #if len(rects) == 0:
    #    return []
    #print rects
    #rects[:,2:] += rects[:,:2]
    #print rects
    # preciso da imagem em escala de cinza para o rastreamento
    return rects,gray

def draw_rects(img, rects, color,listaDeRotulos):
    for x1, y1, w, h in rects:
        cv2.rectangle(img, (x1, y1), (x1+w, y1+h), color, 2)
        cv2.putText(anotado,'R:{}'.format(listaDeRotulos[(x1,y1,w,h)]),(x1,(y1+20)),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(255,255,255),2)
        if len(listaDeRotulos[(x1,y1,w,h)])> 1:
            cv2.imshow("Verifica",img)
            cv2.waitKey(0)

#calcula distancia de dois pontos
def calcDistPoints(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    dx = float(x2-x1)
    dy = float (y2 - y1)
    return math.sqrt(dx*dx+dy*dy)

#classe contendo informacoes da face rastreada formada por:
# ttl       - tempo de vida
# identity  - identidade corrente
# prevRect  - posicao da ultimo rectangulo
# prevFrame - contagem do frame corrente
# Adicionais center, e maxDist
class TrackedFace():
    def __init__(self,currentRect,currentFrame,identity,maxttl):
        # tempo maximo de vida do rastreamento
        self.maxttl = maxttl
        # tempo de vida em numero de frames
        self.ttl = self.maxttl
        # identidade corrente atribuida na stream
        self.identity = identity
        self.prevRect = currentRect
        self.prevFrame = currentFrame
        x1,y1,x2,y2=currentRect
        # calculo de centro da face localizada
        self.center = ((x1+x2)/2,(y1+y2)/2)
        # distancia maxima esperada para localizar o mesmo rosto no tempo de vida
        self.maxDist =  (x2-x1)/4

    # verifica se o retangulo do frame corrente refere-se a face e retorna verdadeiro se afirmativo
    def verify(self,rect,currentFrame):
        x1,y1,x2,y2 = rect
        newcenter = ((x1+x2)/2,(y1+y2)/2)
        dist = calcDistPoints(self.center,newcenter)
        if self.maxDist > dist:
            if self.prevFrame == currentFrame:
                print "A identidade {} ja foi atribuida no frame corrente".format(self.identity)
            else:
                self.prevRect=rect
                self.prevFrame=currentFrame
                # reinicia contagem do ciclo de vida da face
                self.ttl = self.maxttl
            return True
        return False

    # faz update e se verdade mantem para o proximo ciclo
    def update(self,currentFrame):
        if self.prevFrame != currentFrame:
            self.ttl -= 1
        if self.ttl == 0:
            return False
        else:
            x1,y1,x2,y2 = self.prevRect
            self.center = ((x1+x2)/2,(y1+y2)/2)
            self.maxDist =  (x2-x1)/4
            return True

# classe para controlar tracking das faces
class TrackingFaces():
    def __init__(self,maxttl=5):
        #lista de faces rastreadas
        self.listTrackedFaces = []
        #contador de identidade
        self.countIdentity = 0
        # tempo maximo de rastreamento em frames
        self.maxttl = maxttl

    #processamento atraves de lista de retangulos no formato posicao incial largura altura
    def updateFromRectWH(self,currentFrame,rects,gray):
        newlistTrackedFaces = []
        listLabedFaces = {}
        for x1,y1,w,h in rects:
            # verificador para nova identidade se nao encontrar uma identidade
            isNotFound = True
            listLabedFaces[(x1,y1,w,h)]=[]
            for tf in self.listTrackedFaces:
                if tf.verify((x1,y1,x1+w,y1+h),currentFrame):
                    # processa caso encontre uma identidade na lista
                    isNotFound = False
                    listLabedFaces[(x1,y1,w,h)].append(tf.identity)
            if isNotFound:
                # processa caso nao encontre correspondente anterior
                newlistTrackedFaces.append(TrackedFace((x1,y1,x1+w,y1+h),currentFrame,self.countIdentity,self.maxttl))
                listLabedFaces[(x1,y1,w,h)].append(self.countIdentity)
                self.countIdentity += 1

        for tf in self.listTrackedFaces:
            # para atualizar ciclo de vida das faces rastreadas
            if tf.update(currentFrame):
                #processa se ciclo de vida da face rastreada nao espirou
                newlistTrackedFaces.append(tf)
            else:
                print "Individuo {} foi removido da lista de rastreamento".format(tf.identity)
        # atualiza a lista de faces rastreadas
        self.listTrackedFaces = newlistTrackedFaces
        return listLabedFaces


if __name__=="__main__":
    args = parser.parse_args()
    print "Carregando lista de arquivos de videos em {}".format(args.dir)
    arquivos = os.listdir(args.dir)
    cascade = cv2.CascadeClassifier(args.cascade)

    # objeto para controle de rastreamento de faces
    trackingF = TrackingFaces()

    for arq in arquivos:
        print "Processando arquivo {}".format(arq.split('-'))
        cap = cv2.VideoCapture(args.dir+'/'+arq)
        temframe=True
        conta = 0
        while(temframe):
            t = clock()
            temframe, frame = cap.read()
            if temframe:
                conta += 1
                anotado=frame.copy()
                print "Frame {}".format(conta)
                rects,grayimg =detect(frame, cascade)

                listaDeRotulos=trackingF.updateFromRectWH(conta, rects,grayimg)

                if len(rects)>0:
                    draw_rects(anotado,rects,(0,0,255),listaDeRotulos)
                    x1,y1,x2,y2 = rects[0]
                    cv2.putText(anotado,'F:{}'.format(conta),(0,(20)),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(255,255,255),1)
                    #cv2.waitKey(5)
                cv2.imshow("Anotado",anotado)
                dt = clock() - t
                print "Total {} ms".format(dt*1000)
                cv2.waitKey(5)

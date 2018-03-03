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

from common import clock
import cv2

#import RepUtil

import numpy as np
import math

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str,
                    help="Video a capturar",
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

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

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

def draw_rects(img, rects, color,listaDeRotulos=None):
    for x1, y1, w, h in rects:
        cv2.rectangle(img, (x1, y1), (x1+w, y1+h), color, 2)
        if listaDeRotulos is not None:
            cv2.putText(anotado,'R:{}'.format(listaDeRotulos[(x1,y1,w,h)]),(x1,(y1+20)),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(255,255,255),2)
            if len(listaDeRotulos[(x1,y1,w,h)])> 1:
                cv2.imshow("Verifica",img)
                #cv2.waitKey(0)

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
    def __init__(self,currentRect,currentFrame,identity,maxttl,grayimg=None):
        # tempo maximo de vida do rastreamento
        self.maxttl = maxttl
        # tempo de vida em numero de frames
        self.ttl = self.maxttl
        # identidade corrente atribuida na stream
        self.identity = identity
        self.prevRect = currentRect
        self.prevFrame = currentFrame
        self.updateVAI(grayimg)
        #print "Criado objeto TrackedFace com identidade {} na posicao {}".format(self.identity,self.prevRect)
        #listagem de faces para uso em debug
        self.listHistory = []
        #para amostragem
        x1,y1,x2,y2=self.prevRect
        imgh = grayimg[y1:y2,x1:x2].copy()
        w = x2 - x1
        if w > 120:
            fator = 120.0/w
            imgh = cv2.resize(imgh,(0,0),fx=fator,fy=fator)
        self.listHistory.append(imgh)
        self.contaSamp = 1

    # para atualizar variaveis adicionais e controle de rastreamento por Lucas-Kanade
    def updateVAI(self,grayimg,fracmx=4):
        x1,y1,x2,y2=self.prevRect
        # calculo de centro da face localizada
        self.center = ((x1+x2)/2,(y1+y2)/2)
        # distancia maxima esperada para localizar o mesmo rosto no tempo de vida
        self.maxDist =  (x2-x1)/fracmx
        self.initLK(grayimg)
        # variavel para controlar atualizacao por rastreamento
        self.tkLK = False



    def initLK(self,grayimg):
        #obtem forma da imagem na escala de cinza
        #wib,heb = grayimg.shape
        #print "Trantando na area {}".format(self.prevRect)
        #gerar enquadramento para uso no rastreamento
        xn1,yn1,xn2,yn2 = self.prevRect
        w = xn2 - xn1
        dx = w / 6
        xn1 += dx
        xn2 -= dx
        yn2 = (yn1+yn2)/2

        #= RepUtil.novoEquadR(self.prevRect, wib, heb)
        #print "Novo enquadramento inicial : ({},{}) ({},{}) para identidade {}".format(xn1,yn1,xn2,yn2,self.identity)
        #area de interesse para aplicar lucas kanade
        self.oldgray = grayimg[yn1:yn2,xn1:xn2].copy()
        #extrai os pontos para rastrear
        self.p0 = cv2.goodFeaturesToTrack(self.oldgray, mask = None, **feature_params)

    # compara se encontram na mesma posicao de detecao e remove o mais antigo para criar um novo grupo de identificacao
    def isSamePos(self,outro):
        x1,y1,x2,y2     = self.prevRect
        xo1,yo1,xo2,yo2 = outro.prevRect
        if x1 == xo1 and y1 == yo1 and x2 == xo2 and y2 == yo2:
            if self.identity < outro.identity:
                return self
            else:
                return outro

    # verifica se o retangulo do frame corrente refere-se a face e retorna verdadeiro se afirmativo
    def verify(self,rect,currentFrame):
        x1,y1,x2,y2 = rect
        newcenter = ((x1+x2)/2,(y1+y2)/2)
        dist = calcDistPoints(self.center,newcenter)
        if self.maxDist > dist:
            if self.prevFrame == currentFrame:
                print "A identidade {} na posiciao {} confrita com {}".format(self.identity,self.prevRect,rect)
                #atualiza a regiao para proposito do rastreamento
                self.prevRect=rect
                # gera contagem para eliminar duplo rastreamento
                self.ttl = 1
            else:
                self.prevRect=rect
                self.prevFrame=currentFrame
                # reinicia contagem do ciclo de vida da face
                self.ttl = self.maxttl
            return True
        return False

    # processa a existencia da face no frame corrente
    def trackLK(self,currentFrame,grayimg):
        #pode ser rastreavel dentro de 3 frames ap 30 fps!!
        if self.p0 is None:  # and (currentFrame-self.prevFrame) > 16
            # se pontos anteriores forem zero retorna sem procesar
            return

        #gerar enquadramento para uso no rastreamento
        x1,y1,x2,y2 = self.prevRect
        w = x2 - x1
        dx = w / 6
        xn1 = x1 + dx
        xn2 = x2 - dx
        yn1 = y1
        yn2 = (y1+y2)/2
        #= RepUtil.novoEquadR(self.prevRect, wib, heb)
        #print "Novo enquadramento ({},{}) ({},{}) para identidade {}".format(xn1,yn1,xn2,yn2,self.identity)
        newgray= grayimg[yn1:yn2,xn1:xn2].copy()
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.oldgray, newgray, self.p0, None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = self.p0[st==1]

        #pr1 = np.array((xn1,yn1))
        conta = 0
        #soma  = 0
        somax = 0
        somay = 0
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            #na,nb = np.array((a,b))+pr1
            #nc,nd = np.array((c,d))+pr1
            #na = int(na)
            #nb = int(nb)
            #nc = int(nc)
            #nd = int(nd)
            #cv2.line(grayimg, (na,nb),(nc,nd), 0 , 2)
            #cv2.circle(grayimg,(na,nb),5,255,-1)
            conta += 1
            #soma  += calcDistPoints((na,nb),(nc,nd))
            somax +=  a - c
            somay +=  b - d
        #considerando que tenha mais de 4 pontos de controle
        if conta > 4:
            #media  = soma/conta
            mediax = int(somax/conta)
            mediay = int(somay/conta)
            #print "Individuo {} com pontos {}".format(self.identity,self.prevRect)
            #print "Antes: {} em {}".format(self.prevRect,self.identity)
            x1 = x1+mediax
            x1 = x1 if x1 > 0 else 0
            y1 = y1+mediay
            y1 = y1 if y1 > 0 else 0
            hb,wb = grayimg.shape
            x2 = x2+mediax
            x2 = x2 if x2 < wb else wb
            y2 = y2+mediay
            y2 = y2 if y2 < hb else hb
            self.prevRect = (x1,y1,x2,y2)
            #print "Depois: {} em {}".format(self.prevRect,self.identity)
            #print "para pontos {}".format(self.prevRect)
            # seta para atualizar
            # considera para tracking se as posicoes x e y moverem
            #if mediax > 0 and mediay > 0:

            self.tkLK = True
            #disg   = math.sqrt((mediax*mediax)+(mediay*mediay))
            #ratiog = media/disg
            #print "Deslocou {} pp no individuo {} com {} ponto com vetor de deslocamento ({},{}) e razao de pp por g {}".format(media,self.identity,conta,mediax,mediay,ratiog)
        # Now update the previous frame and previous points
        #self.oldgray = newgray
        #self.p0 = good_new.reshape(-1,1,2)


    # faz update e se verdade mantem para o proximo ciclo
    def update(self,currentFrame,grayimg=None): #,fracmx=4
        #tratamento quando em modo de rastreio
        if self.prevFrame != currentFrame:
            self.ttl -= 1
            if self.ttl == 0:
                return False
            if self.tkLK:
                #nao diminui ttl pois apresenta possibilidade de rastreo
                #print "Atualizado por track objeto TrackedFace com identidade {} na posicao {} com ttl {}".format(self.identity,self.prevRect,self.ttl)
                self.updateVAI(grayimg) #,fracmx
                return True

        if self.ttl == 0:
            return False
        else:
            #x1,y1,x2,y2 = self.prevRect
            #self.center = ((x1+x2)/2,(y1+y2)/2)
            #self.maxDist =  (x2-x1)/fracmx
            self.updateVAI(grayimg) #,fracmx
            #print "Remanecente objeto TrackedFace com identidade {} na posicao {} com ttl {}".format(self.identity,self.prevRect,self.ttl)
            #preenchendo lista de hisotorico
            x1,y1,x2,y2 = self.prevRect
            imgh = grayimg[y1:y2,x1:x2].copy()
            w = abs(x2 - x1)
            if w > 119:
                fator = 119.0/w
                #print "{};{};{};{} -> {};{}".format(x1,y1,x2,y2,w,fator)
                imgh = cv2.resize(imgh,(0,0),fx=fator,fy=fator)
                #imgh = imgr            
            h,w = imgh.shape
            if h > 119:
                fator = 119.0/h
                #print "{};{};{};{} -> {};{}".format(x1,y1,x2,y2,w,fator)
                imgh = cv2.resize(imgh,(0,0),fx=fator,fy=fator)
                #imgh = imgr
                
            if len(self.listHistory) == 10:
                self.listHistory[self.contaSamp%10] = imgh
            else:
                self.listHistory.append(imgh)
            self.contaSamp += 1
            return True

# classe para controlar tracking das faces
# maxttl - maximo numero de frames para tempo de rastreamento em 30fps o valor 10 represetenta 330ms
# maxttl - maximo numero de frames para tempo de rastreamento em 30fps o valor 15 represetenta 500ms
class TrackingFaces():
    def __init__(self,maxttl=15):
        #lista de faces rastreadas
        self.listTrackedFaces = []
        #contador de identidade
        self.countIdentity = 0
        # tempo maximo de rastreamento em frames
        self.maxttl = maxttl

    #processamento atraves de lista de retangulos no formato posicao incial largura altura
    def updateFromRectWH(self,currentFrame,rects,gray):
        for tf in self.listTrackedFaces:
            tf.trackLK(currentFrame,gray)

        newlistTrackedFaces = []
        listToRemoveDup = []
        listLabedFaces = {}
        for x1,y1,w,h in rects:
            # verificador para nova identidade se nao encontrar uma identidade
            isNotFound = True
            # caso encontre anterior
            prevTf = None
            listLabedFaces[(x1,y1,w,h)]=[]
            for tf in self.listTrackedFaces:
                if tf.verify((x1,y1,x1+w,y1+h),currentFrame):
                    # atualiza os pontos de rastreamento na face localizada
                    tf.initLK(gray)
                    if not isNotFound:
                        print "Anterior {} na posicao {} confrita com {} na posicao {}".format(prevTf.identity,prevTf.prevRect,tf.identity,tf.prevRect)
                        listToRemoveDup.append(prevTf) #tf.isSamePos(prevTf))
                        # remove tambem a atual da lista
                        listToRemoveDup.append(tf)
                    # processa caso encontre uma identidade na lista
                    isNotFound = False
                    prevTf = tf
                    listLabedFaces[(x1,y1,w,h)].append(tf.identity)
            if isNotFound:
                # processa caso nao encontre correspondente anterior
                newlistTrackedFaces.append(TrackedFace((x1,y1,x1+w,y1+h),currentFrame,self.countIdentity,self.maxttl,grayimg=gray))
                listLabedFaces[(x1,y1,w,h)].append(self.countIdentity)
                self.countIdentity += 1
        for tf in self.listTrackedFaces:
            # para atualizar ciclo de vida das faces rastreadas
            if tf.update(currentFrame,grayimg=gray) and tf not in listToRemoveDup:
                #processa se ciclo de vida da face rastreada nao espirou
                newlistTrackedFaces.append(tf)
            else:
                print "Individuo {} foi removido da lista de rastreamento".format(tf.identity)
        # atualiza a lista de faces rastreadas
        self.listTrackedFaces = newlistTrackedFaces
        return listLabedFaces

    #obter a lista de faces em track state
    def getOnTrackFaces(self,currentFrame):
        listRects = []
        listaNomes = {}
        for tf in self.listTrackedFaces:
            if tf.prevFrame != currentFrame:
                x1,y1,x2,y2 = tf.prevRect
                w = x2 - x1
                h = y2 - y1
                listRects.append((x1,y1,w,h))
                listaNomes[(x1,y1,w,h)]=[]
                listaNomes[(x1,y1,w,h)].append((tf.identity,tf.ttl))

        return listRects,listaNomes

    #obter historico de imagens de face em rastreo
    def drawSampleTrack(self):

        #numero de linhas de amostras
        n = len(self.listTrackedFaces)
        if n < 1:
            return
        # imagem
        largeimg = np.zeros((120*n+10,1200+10),np.uint8)
        ypos = 0
        for tf in self.listTrackedFaces:
            xpos = 0
            for img in tf.listHistory:
                h,w = img.shape
                #print "Origem: {}".format(img.shape)    
                largeimg[ypos:ypos+h,xpos:xpos+w]=img
                xpos += 120
            cv2.putText(largeimg,str(tf.identity),(0,ypos+20),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0),3)
            ypos += 120
        cv2.imshow("Sample",largeimg)
        #cv2.waitKey(0)



if __name__=="__main__":
    args = parser.parse_args()
    print "Carregando video em {}".format(args.video)
    cascade = cv2.CascadeClassifier(args.cascade)

    # objeto para controle de rastreamento de faces
    trackingF = TrackingFaces()
    cap = cv2.VideoCapture(args.video)
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
                #cv2.imshow("Cinza",grayimg)

                cv2.putText(anotado,'F:{}'.format(conta),(0,(20)),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(255,255,255),1)
                if len(rects)>0:
                    draw_rects(anotado,rects,(0,0,255),listaDeRotulos=listaDeRotulos)
                    x1,y1,x2,y2 = rects[0]

                rectst,nomest=trackingF.getOnTrackFaces(conta)

                if len(rectst)>0:
                    draw_rects(anotado,rectst,(0,255,255),listaDeRotulos=nomest)

                    #cv2.waitKey(0)
                cv2.imshow("Anotado",anotado)
                # apresentado amostragem
                trackingF.drawSampleTrack()
                dt = clock() - t
                print "Total {} ms".format(dt*1000)
                cv2.waitKey(1)
                # teste de rotacao
                (h, w) = frame.shape[:2]
                center = (w / 2, h / 2)
                #center = (0 , 0)
                # rotate the image by 180 degrees
                #M = cv2.getRotationMatrix2D(center, 90, 1.0)
                #rotated = cv2.warpAffine(frame, M, (h, w))
                #cv2.imshow("rotated", rotated)

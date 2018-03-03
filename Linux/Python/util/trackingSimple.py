import cv2
import numpy as np
from miscsolve import calcDistPoints,novoEquadR

from os.path import join
import thread
from datetime import datetime
from cmath import rect
from align_dlib import AlignDlib
align = AlignDlib(join("..","..","data","models","dlib","shape_predictor_68_face_landmarks.dat"))

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.025,
                       minDistance = 5,
                       blockSize = 5 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

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
        # identidade corrente atribuida na stream
        self.identity = identity
        self.prevRect = currentRect
        self.prevFrame = currentFrame
        # limite minimo de movimento para considerar rastreavel em termis de pixels
        #self.limMov = limMov
        self.updateVAI()

        #print "Criado objeto TrackedFace com identidade {} na posicao {}".format(self.identity,self.prevRect)
        #listagem de faces para uso em debug
        self.listHistory = []
        self.listHistory.append(currentRect)
        #Contagem de amostras
        self.contaSamp = 0
        # controle para atualizar quando executado update
        self.readyUp = False

    # para atualizar variaveis adicionais e controle de rastreamento por Lucas-Kanade
    #fracmx o default era 4 teste com 2
    def updateVAI(self):
        idx,(x1,y1,x2,y2)=self.prevRect
        # calculo de centro da face localizada
        self.center = ((x1+x2)/2,(y1+y2)/2)
        # distancia maxima esperada para localizar o mesmo rosto no tempo de vida
        #self.maxDist =  (x2-x1)/fracmx


    # verifica se o retangulo do frame corrente refere-se a face e retorna verdadeiro se afirmativo
    def verify(self,rect,currentFrame):
        x1,y1,x2,y2 = rect
        idx,(xl1,yl1,xl2,yl2) = self.prevRect
        w1 = xl2-xl1
        w2 = x2 - x1        
        maxDist = (w1+w2)/6.0
        newcenter = ((x1+x2)/2,(y1+y2)/2)
        dist = calcDistPoints(self.center,newcenter)
        if dist < maxDist and (currentFrame - self.prevFrame) < self.maxttl:
            return True
        return False

    def registry(self,rect,currentFrame):
        self.readyUp = True
        self.temprect = rect
        self.tempidxFrame = currentFrame
        
    # faz update e se verdade mantem para o proximo ciclo
    def update(self): #,fracmx=4
        #tratamento quando em modo de rastreio unificado  com ou sem atualizacao do tratador de track LK
        if self.readyUp:
            self.listHistory.append(self.temprect)
            self.prevFrame = self.tempidxFrame
        self.readyUp = False


# classe para controlar tracking das faces
# maxttl - maximo numero de frames para tempo de rastreamento em 30fps o valor 10 represetenta 330ms
# maxttl - maximo numero de frames para tempo de rastreamento em 30fps o valor 15 represetenta 500ms
class TrackingFaces():
    def __init__(self,countIdentity = 0,maxttl=15):
        #lista de faces rastreadas
        self.listTrackedFaces = []
        #contador de identidade
        self.countIdentity = 0
        # tempo maximo de rastreamento em frames
        self.maxttl = maxttl
        # contador para verificar boas amostragens
        self.countGood = 0
        self.conflitIdList = []

    def analisaConflito(self,currentFrame,rect,tf,prevtf):
        idx,(x1,y1,x2,y2) = rect
        w = x2 - x1
        idxa,(xc1,yc1,xc2,yc2) = tf.prevRect
        wc = xc2 - xc1
        idxb,(xp1,yp1,xp2,yp2) = prevtf.prevRect
        wp = xp2 - xp1
        c  = ((x1+x2)/2,(y1+y2)/2)
        cc = ((xc1+xc2)/2,(yc1+yc2)/2)
        cp = ((xp1+xp2)/2,(yp1+yp2)/2)
        
        distc = calcDistPoints(c,cc)
        distp = calcDistPoints(c,cp)
        #print "Conflito ->N {} : {} : {} I: {} : {} C {} : {} : {} L: {} : {} : {} dist: {} : {}".format(currentFrame,tf.prevFrame,prevtf.prevFrame,tf.identity,prevtf.identity,c,cc,cp,w,wc,wp,distc,distp)
        if distc < distp:
            #print "Select {}".format(cc)
            # Desabilita atualizacao do outro 
            tf.registry((idx,rect),currentFrame)
            prevtf.readyUp = False
            return tf
        else:
            #print "Select {}".format(cp)
            # Desabilita atualizacao do outro             
            return prevtf
            
        
        #print "Conflito {}:{}:{} {}:{}:{}".format(currentFrame,tf.prevFrame,prevtf.prevFrame,rect,tf.prevRect,prevtf.prevRect)
        
    #processamento do ratreamento atraves de lista de retangulos no formato posicao incial largura altura
    def updateFromRectWH(self,currentFrame,rects):
        newlistTrackedFaces = []        
        for idx,rect in rects:
            # verificador para nova identidade se nao encontrar uma identidade
            isNotFound = True
            prevtf = None
            for tf in self.listTrackedFaces:
                #determina se os pontos sao desta face 
                if tf.verify(rect,currentFrame):
                    if not isNotFound:
                        if tf.identity < prevtf.identity:
                            self.conflitIdList.append((tf.identity,prevtf.identity))
                        else:
                            self.conflitIdList.append((prevtf.identity,tf.identity))
                        prevtf = self.analisaConflito(currentFrame,(idx,rect),tf,prevtf) 
                    else:
                        # processa caso encontre uma identidade na lista
                        isNotFound = False
                        tf.registry((idx,rect),currentFrame)
                        prevtf = tf
            if isNotFound:
                # processa caso nao encontre correspondente anterior
                newlistTrackedFaces.append(TrackedFace((idx,rect),currentFrame,self.countIdentity,self.maxttl))
                self.countIdentity += 1
        
        remlistTrackedFaces = []
        for tf in self.listTrackedFaces:
            tf.update()
            if currentFrame-tf.prevFrame>self.maxttl:
                remlistTrackedFaces.append(tf)
                
        for tf in remlistTrackedFaces:
            self.listTrackedFaces.remove(tf)
        
        for tf in newlistTrackedFaces:
            self.listTrackedFaces.append(tf)

        return remlistTrackedFaces

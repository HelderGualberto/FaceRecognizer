#!/usr/bin/env python2
# Autor: Roberto Kenji Hiramatsu
#
# Copyright 2016 Universidade de Sao Paulo
#
# Modelo para rastrear face para uso de classificacao sobre a camera
#
# Data inicial: 2016-11-16 com tratador dlib
#             : 2016-11-25 versao com processamento e envio para base do mongodb
#             : 2016-11-29 versao para processamento seletivo e reducao de ocorrencia de deteccao faces fp
#             : 2016-12-06 Ajuste de parametros de rastreamento de LK relacionadas aos pontos de rastreamento e 
#                          area de cobertura de rastreamento  
#             : 2017-03-03 Ajuste para tratar caminho relativo dos arquivos de modelos e todo
#                          Compatibilidade entre 2.4 e 3.1
#
#
# TODO:
#             python modelorastreafaceOnCam3.py --video <fonte de video> --host <mongo host> --port <mongo port> --base <database name> --camfps <fps>
import os

import argparse
import math
from datetime import datetime

from common import clock
import cv2

#import RepUtil

import numpy as np

import thread

from align_dlib import AlignDlib

import libColetaFaceDB

from os.path import join

basedir=join('..','..','data')

#detecta versao do opencv usando 2.4 no linux e 3.x no windows
opencvver = cv2.__version__.split('.')

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str,
                    help="Video a capturar",
                    #default='rtsp://kenji:6qi7j94i@192.168.10.181:554/H264')
                    default='rtsp://admin:B30cd4Ro@192.168.10.180:554/LiveMedia/ch1/Media2')
                    #default='D:\\projetos\\Safety_City\\teste\\teste.mp4')
                    #default="D:\\projetos\\Safety_City\\HBPVR\\TV_Camara-17042016-2354.mts")
                    #default=0) #D:\\temp

#alt2 apresenta melhor desempenho para detecao de face
parser.add_argument('--cascade', type=str,
                    help="cascade haar detector",
                    #default='haarcascades/haarcascade_frontalface_alt_tree.xml')
                    #default='haarcascades/haarcascade_frontalface_alt2.xml')
                    default=join(basedir,'haarcascades','haarcascade_frontalface_alt2.xml')) #D:\\app\\opencv31\\sources\\data\\
                    #default='haarcascades/haarcascade_frontalface_default.xml')
                    #default='haarcascades/haarcascade_frontalface_alt.xml')
                    #default='D:\\app\\opencv31\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml')

parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=join(basedir,"models","dlib", "shape_predictor_68_face_landmarks.dat"))

parser.add_argument('--host', type=str, help="Host to mongodb",
                    default="192.168.10.236")

parser.add_argument('--port', type=int, help="Port of  mongodb",
                    default=37027)

parser.add_argument('--base', type=str, help="Database on  mongodb",
                    #default="huaweicam") # primeira amostragem
                    #default="huaweicam2s")
                    #default="huaweicam3s")
                    #default="huaweicam4s")
                    default="huaweicam13s")

parser.add_argument('--camfps', type=int, help="Frame per Second on cam",
                    default=6)

parser.add_argument('--displayOn',type=bool, help="Display process result",
                    default=False)

args = parser.parse_args()

rotateIm=False

mdb = libColetaFaceDB.MongoConn(url="mongodb://{}:{}".format(args.host,args.port),
                                dbs=args.base)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.025,
                       minDistance = 5,
                       blockSize = 5 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# alocado globalmente o extrador dlib
align = AlignDlib(args.dlibFacePredictor)

#calcula distancia de dois pontos
def calcDistPoints(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    dx = float(x2-x1)
    dy = float (y2 - y1)
    return math.sqrt(dx*dx+dy*dy)

#detecao da face com haar cascade em subretangulos da imagem
def detect(img, cascade,rectpts):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if len(rectpts) < 1:
        return [],gray
    #gray = cv2.equalizeHist(gray)

    rects = []
    for x0,y0,x1,y1 in rectpts:
        # copia da regiao de movimento
        grayroi = gray[y0:y1,x0:x1].copy()
        locrects = cascade.detectMultiScale(grayroi, scaleFactor=1.1, minNeighbors=4, #scaleFactor=1.05 2
                                     minSize=(30, 30))
        for x,y,w,h in locrects:
            rects.append((x0+x,y0+y,w,h))
    return rects,gray

def draw_rects(img, rects, color,listaDeRotulos=None):
    for x1,y1,w,h in rects:
        cv2.rectangle(img, (x1, y1), (x1+w, y1+h), color, 2)
        if listaDeRotulos is not None:
            cv2.putText(anotado,'R:{}'.format(listaDeRotulos[(x1,y1,w,h)]),(x1,(y1+20)),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(255,255,255),2)
            #if len(listaDeRotulos[(x1,y1,w,h)])> 1:
            #    cv2.imshow("Verifica",img)
                #cv2.waitKey(0)

# funcao para enquadramento da area de busca
# rect - pontos P1 e P2 de retangulo
# w    - largura da imagem principal
# h    - altura da imagem principal
# fatorenq - fracao a ser extraida da imagem
def novoEquadR(rect,w,h,fatorenq=6):
    x1,y1,x2,y2 = rect
    dx = (x2-x1)/fatorenq
    x1 = (x1 - dx) if (x1 - dx)>0 else 0
    x2 = (x2 + dx) if (x2 + dx)<w else w
    dy = (y2 -y1)/fatorenq
    y1 = (y1 - dy) if (y1 - dy)>0 else 0
    y2 = (y2 + 2*dy) if (y2 + 2*dy) < h else h
    return x1,y1,x2,y2


#calcula distancia entre dois pontos simples
def calcDisDP(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    dx = float(x2-x1)
    dy = float (y2 - y1)
    return math.sqrt(dx*dx+dy*dy)
#ajusta a imagem baseado em deslocamento de pontos entre frame par e impar
#janela de
frach = 6
def separaOddEven(img):
    [h,w,c] = img.shape
    dx = w/(2*frach)
    par = np.zeros((h/2,2*dx,3),np.uint8)
    impar = np.zeros((h/2,2*dx,3),np.uint8)
    x0 = w/2-dx
    for iy in range(0,h/2-h/6):
        par[iy,:]=img[2*iy,x0:x0+2*dx]
        impar[iy,:]=img[2*iy+1,x0:x0+2*dx]
    #escala de sinca do par
    gpar = cv2.cvtColor(par,cv2.COLOR_BGR2GRAY)
    #selecao de pontos no frame
    p0 = cv2.goodFeaturesToTrack(gpar, mask = None, **feature_params)
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    gimpar = cv2.cvtColor(impar,cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    try:
        p1, st, err = cv2.calcOpticalFlowPyrLK(gpar, gimpar, p0, None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        #print "Pontos de track",p0
        # Create a mask image for drawing purposes
        mask = np.zeros_like(par)

        # draw the tracks
        contap = 0
        soma = 0
        somax = 0
        somay = 0
        somaqx = 0
        somaqy = 0
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            contap += 1
            soma += calcDisDP((a,b),(c,d))
            vx = float(c - a)
            somax += vx
            somaqx += vx * vx
            vy = float(d -b)
            somay += vy
            somaqy += vy * vy
            #print "Posicoes deslocadas",(a,b),(c,d)," vetor: ",(c-a,d-b)
            #cv2.line(impar, (a,b),(c,d), color[i].tolist(), 1)
            #cv2.circle(impar,(a,b),5,color[i].tolist(),-1)
            #cv2.imshow("Par",par)
            #cv2.imshow("Impar",impar)

        if contap >1 :
            media = soma / float(contap)
            mediax =  somax / float(contap)
            varix = (somaqx - somax*somax/contap)/(contap-1)
            variy = (somaqy - somay*somay/contap)/(contap-1)
            mediay =  somay / float(contap)
            mediay += 0.25 if mediay < 0 else -0.25
            #print "Media de distancia deslocada de ",media," em  ",contap," pontos e vet ",(mediax,mediay)," x des ",round(mediax)," y des ",round(mediay)," vari ",(varix,variy)
            if varix < 2.25 and variy < 1.25 and contap > 2:
                #
                if (abs(mediax) >= 0.5  or abs(mediay) >= 1.0):
                    if abs(mediay) < 2.0 and abs(mediax) < 3.5:
                        desloca = int(round(mediax))
                        ajustado = np.zeros((h,w,3),np.uint8)
                        deslocay = int(round(mediay))
                        if deslocay < 0:
                            # quando o quadro impar esta mais acima
                            yi = 0
                            yf = h - 2*deslocay-8
                        else:
                            yi = 2*deslocay
                            yf = h-8
                        for iy in range(yi,yf):
                            if (iy%2) == 0:
                                ajustado[iy,:]=img[iy,:]
                            else:
                                if abs(desloca) == 0:
                                    ajustado[iy,desloca:] = img[iy,:]
                                else:
                                    if desloca > 0:
                                        ajustado[iy,desloca:] = img[iy,:-desloca]
                                    else:
                    #print iy,desloca,h,w,yi,yf
                                        ajustado[iy,:desloca] = img[iy-2*deslocay,-desloca:]
                        #cv2.imshow("Original",img.copy())
                        #cv2.imshow("Ajustado",ajustado)
                        #cv2.waitKey(0)
                        return ajustado,True
                    else:
                        #muito deslocamento para ajustar adequadamente a imagem
                        return img,False
                return img,True
        else:
            # tem menos de um ponto de tracking em movimento
            return img,True
    except cv2.error:
        print "Se pontos de deslocamento "
    return img,False


#classe contendo informacoes da face rastreada formada por:
# ttl       - tempo de vida
# identity  - identidade corrente
# prevRect  - posicao da ultimo rectangulo
# prevFrame - contagem do frame corrente
# Adicionais center, e maxDist
class TrackedFace():
    def __init__(self,currentRect,currentFrame,identity,maxttl,grayimg=None,limMov=0.05):
        # tempo maximo de vida do rastreamento
        self.maxttl = maxttl
        # tempo de vida em numero de frames
        self.ttl = self.maxttl
        # identidade corrente atribuida na stream
        self.identity = identity
        self.prevRect = currentRect
        self.prevFrame = currentFrame
        # limite minimo de movimento para considerar rastreavel em termis de pixels
        self.limMov = limMov
        self.updateVAI(grayimg)
        # variavel para avisar se o objeto em questao esta vivo!!!
        self.goodOne = True
        #print "Criado objeto TrackedFace com identidade {} na posicao {}".format(self.identity,self.prevRect)
        #listagem de faces para uso em debug
        self.listHistory = []
        #Contagem de amostras
        self.contaSamp = 0

    # para atualizar variaveis adicionais e controle de rastreamento por Lucas-Kanade
    #fracmx o default era 4 teste com 2
    def updateVAI(self,grayimg,fracmx=2):
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
        heb,wib = grayimg.shape
        #print "Trantando na area {}".format(self.prevRect)
        #gerar enquadramento para uso no rastreamento
        xn1,yn1,xn2,yn2 = self.prevRect
        w = xn2 - xn1
        dx = w / 6
        xn1 -= dx
        xn1 = xn1 if xn1 >0 else 0
        xn2 += dx
        xn2 = xn2 if xn2 < wib else wib
        #yn2 = (yn1+yn2)/2

        #= RepUtil.novoEquadR(self.prevRect, wib, heb)
        #print "Novo enquadramento inicial : ({},{}) ({},{}) para identidade {}".format(xn1,yn1,xn2,yn2,self.identity)
        #area de interesse para aplicar lucas kanade
        self.oldgray = grayimg[yn1:yn2,xn1:xn2].copy()
        #extrai os pontos para rastrear
        self.p0 = cv2.goodFeaturesToTrack(self.oldgray, mask = None, **feature_params)
        #self.drawGP()
        
    def drawGP(self):
        img = cv2.cvtColor(self.oldgray,cv2.COLOR_GRAY2BGR)
        
        if self.p0 is None:
            return
        
        corners = np.int0(self.p0)
        
        for i in corners:
            x,y = i.ravel()
            cv2.circle(img,(x,y),3,(0,0,255),-1)
        cv2.putText(img,'{}'.format(self.identity),(0,(20)),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,0,255),1)
        if args.displayOn:
            cv2.imshow("Good",img)
            cv2.waitKey(0)

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
            #print "Retornado pois p0 vazio???"
            return
        
        heb,wib = grayimg.shape
        #gerar enquadramento para uso no rastreamento
        x1,y1,x2,y2 = self.prevRect
        w = x2 - x1
        dx = w / 6
        xn1 = x1 - dx
        xn1 = xn1 if xn1 > 0 else 0
        xn2 = x2 + dx
        xn2 = xn2 if xn2 < wib else wib
        #self.drawGP()
        #yn1 = y1
        #yn2 = (y1+y2)/2
        #= RepUtil.novoEquadR(self.prevRect, wib, heb)
        #print "Novo enquadramento ({},{}) ({},{}) para identidade {}".format(xn1,yn1,xn2,yn2,self.identity)
        newgray= grayimg[y1:y2,xn1:xn2].copy()
        
                        
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.oldgray, newgray, self.p0, None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = self.p0[st==1]

        #pr1 = np.array((xn1,yn1))
        conta = 0
        soma  = 0
        somax = 0
        somay = 0
        #newgrayR = newgray.copy()
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            #print "Pontos (a,b) = ({},{}) (c,d) = ({},{})".format(a,b,c,d)
            #na,nb = np.array((a,b))+pr1
            #nc,nd = np.array((c,d))+pr1
            #na = int(a)
            #nb = int(b)
            #nc = int(nc)
            #nd = int(nd)
            #cv2.line(grayimg, (na,nb),(nc,nd), 0 , 2)
            #cv2.circle(newgrayR,(na,nb),5,255,-1)
            conta += 1
            soma  += calcDistPoints((a,b),(c,d))
            somax +=  a - c
            somay +=  b - d
        
        #=======================================================================
        # if newgray is not None:
        #     h,w = newgray.shape
        #     if w > 0 and h > 0:
        #         cv2.putText(newgrayR,'{}'.format(self.identity),(0,(20)),cv2.FONT_HERSHEY_SIMPLEX,
        #                         0.6,(255,255,255),1)
        #         cv2.imshow("New",newgray)
        #         cv2.imshow("old",self.oldgray)
        #         cv2.waitKey(0)
        #=======================================================================
        
        #para tratamento de regioes de nao face
        if conta > 1:
            media  = soma/conta
            #print "Tem media: {} em {} pontos e largura {} do individuo {}".format(media,conta,w,self.identity)
            if media < self.limMov: #
                # eliminando ponto por estar muto abaixo do esperado!!!
                self.ttl = 1
                self.goodOne = False
                #print "Retornado pois tem pouco movimento???"
                return
        #considerando que tenha mais de 4 pontos de controle
        if conta > 4:
            #media  = soma/conta
            mediax = int(somax/conta)
            mediay = int(somay/conta)
            #print "Individuo {} com pontos {}".format(self.identity,self.prevRect)
            #print "media:x {} y {} em {}".format(mediax,mediay,self.identity)
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
        #else:
        #    print "Nao processado media pois tem poucos pontos"
            # Now update the previous frame and previous points
        #self.oldgray = newgray
        #self.p0 = good_new.reshape(-1,1,2)


    # faz update e se verdade mantem para o proximo ciclo
    def update(self,currentFrame,grayimg=None): #,fracmx=4
        #tratamento quando em modo de rastreio unificado  com ou sem atualizacao do tratador de track LK
        if self.prevFrame != currentFrame:
            self.ttl -= 1
            if self.ttl == 0:
                return False
            if self.tkLK:
                #nao diminui ttl pois apresenta possibilidade de rastreo
                #print "Atualizado por track objeto TrackedFace com identidade {} na posicao {} com ttl {}".format(self.identity,self.prevRect,self.ttl)
                self.updateVAI(grayimg) #,fracmx
                return True

        # se nao for bom entao deve ser eliminado pois tem pouco movimento
        if not self.goodOne:
            return False
        #if self.ttl == 0:
        #    return False
        #else:
            #x1,y1,x2,y2 = self.prevRect
            #self.center = ((x1+x2)/2,(y1+y2)/2)
            #self.maxDist =  (x2-x1)/fracmx
        self.updateVAI(grayimg) #,fracmx

        return True

    # coleta a amostra da imagem de ratreamento
    # img          - imagem colorida em tamanho original
    # currentframe - contador do frame corrente
    # fator        - reducao aplicada para detecao da face
    # maxWidth     - largura maxima da imagem coletada
    # minWidth     - largura minima da face
    # faceAli      - processa alinhamento da face
    # retorna (isCap,seq,cropimg) -
    #          isCap   - indicador que a imagem foi coletada
    #          seq   - contagem do coletor
    #          cropimg - imagem capturada
    def collectSample(self,img,currentframe,faceAli=False,fator=1.0,maxWidth=180.0,minWidth=107.0): #era maxWidth=120.0
        # Obtem tamaho original na imagem
        x1,y1,x2,y2 = (np.array(self.prevRect)/fator).astype(int)
        # obtem o valor da largura da imagem
        ws = abs(x2-x1)
        # retorna se for menor que minWidth. Isto e, a imagem esta abaixo no nivel para rede com 96
        #print "ws:{}".format(ws)
        if ws < minWidth:
            return False,self.contaSamp,None
        #fator de redimencionamento
        fatorS = 1.0  if ws < maxWidth else maxWidth/float(ws)

        h,w,c = img.shape
        # obtendo os valores da janela de recorte a ser usada no dlib
        x1,y1,x2,y2 = novoEquadR((x1,y1,x2,y2),w,h)
        #recortando a regiao da imagem
        sample = img[y1:y2,x1:x2].copy()
        
        
        # redimencionado para faixa adequada para processamento na dlib
        sample = cv2.resize(sample,(0,0),fx=fatorS,fy=fatorS)

        #processamento adicional para desentlelacar
        #sample,ret = separaOddEven(sample)
        #if ret:
        #    cv2.imshow("Ajusta sample",sample)
        #else:
        #if not ret:
        #    return False,self.contaSamp,None
        # processa dlib
        bb = align.getLargestFaceBoundingBox(sample)
        #processa para separar somente rosto mesmo extraido de haar
        if bb is None:
            return False,self.contaSamp,None

        wb = abs(bb.right()-bb.left())
        hsr,wsr,c = sample.shape
        if float(wb)/float(wsr) < 0.6:
            print "Recover image sample because face in sample image is small"

            xe1 = x1+int(bb.left()/fatorS)
            ye1 = y1+int(bb.top()/fatorS)
            xe2 = x1+int(bb.right()/fatorS)
            ye2 = y1+int(bb.bottom()/fatorS)
            x1,y1,x2,y2 = novoEquadR((xe1,ye1,xe2,ye2),w,h)
            #recortando a regiao da imagem
            sample = img[y1:y2,x1:x2].copy()
            fatorS = 1.0
            wb= abs(x2 - x1)
            if faceAli:
                bb = align.getLargestFaceBoundingBox(sample)

        if faceAli:
            if bb is not None:
                rwi = (wb/(ws*fatorS))
                if  rwi < 0.7 or rwi >1.4:
                    return False,self.contaSamp,None
                landmarks = align.findLandmarks(sample, bb)
                sample = align.align(96, sample, bb,
                                      landmarks=landmarks,
                                      landmarkIndices=align.OUTER_EYES_AND_NOSE)
            else:
                #se nao encontrar face entao deve ser desconsiderado
                return False,self.contaSamp,None

        #arq = "samp_{:04d}_{:02d}_{:06d}.jpg".format(self.identity,self.contaSamp,currentframe)
        #cv2.imwrite(arq,sample)
        #listagem de faces para uso em debug
        self.listHistory.append((currentframe,sample))
        #cv2.putText(sample,str(self.identity),(0,20),cv2.FONT_HERSHEY_SIMPLEX,
                                #0.6,(0,255,255),1)
        #cv2.imshow("Sample",sample)
        #cv2.waitKey(0)
        #Contagem de amostras
        self.contaSamp += 1
        # retorna a contagem atual de amostras
        return  True,self.contaSamp,sample

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

    #processamento do ratreamento atraves de lista de retangulos no formato posicao incial largura altura
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
                if tf.contaSamp > 3:
                    self.countGood += 1
                    print "Contado {} faces com mais de tres amostras".format(self.countGood)
                #print "Individuo {} foi removido da lista de rastreamento".format(tf.identity)
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
            print "Retorna pois nao tem imagens armazenadas"
            return

        # imagem
        largeimg = np.zeros((120*n+120,1200+120,3),np.uint8)
        ypos = 0

        for tf in self.listTrackedFaces:
            if tf.contaSamp < 2:
                continue
            xpos = 0
            nim = len(tf.listHistory)
            print "nim: {}".format(nim)
            idxc = 0
            for idx,(cframe,img) in enumerate(tf.listHistory):
                h,w,c = img.shape
                #print "Origem: {}".format(img.shape)
                if (nim-idxc) < 10:
                    largeimg[ypos:ypos+h,xpos:xpos+w]=img
                    cv2.putText(largeimg,str(cframe),(xpos,ypos+60),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),1)
                    xpos += 120
                else:
                    tf.listHistory.remove((cframe,img))
                idxc += 1
                #print "idxc:{}".format(idxc)
            
            cv2.putText(largeimg,str(tf.identity),(0,ypos+20),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),3)

            ypos += 120
        if args.displayOn:
            if ypos > 0:
                cv2.imshow("Sample",largeimg)
            #cv2.waitKey(0)
        #cv2.waitKey(0)

    #coletor de imagens
    # currentFrame - contagem do frame atual
    # fator        - reducao de escala usada para melhorar o desempenho de localizacao de face na imagem
    #                e referencia para extracao da face em escala na imagem original.
    def collectCurrentSamp(self,currentFrame,img,fator,mdb=None,base="camara",faceAlinhada = False):
        retop = False
        for tf in self.listTrackedFaces:
            isCap,seq,cropimg = tf.collectSample(img,currentFrame,faceAli=faceAlinhada,fator=fator)
            if mdb is not None:
                if seq > 1: # a partir de duas faces
                    # somente processar se for a terceira imagem da sequencia
                    if seq == 2 and isCap:
                        #armazenar a imagem anteriores da sequencia
                        #gravaIm(self,img,base,seq,frame,datahora,depessoa,alinhado)
                        for idx,(cframe,sample) in enumerate(tf.listHistory):
                            thread.start_new_thread(mdb.gravaIm,(sample, base, idx, cframe, datetime.now(), tf.identity, faceAlinhada))
                            retop = True
                    else :
                        if isCap:
                            thread.start_new_thread(mdb.gravaIm,(cropimg, base, seq, currentFrame, datetime.now(), tf.identity, faceAlinhada))
                            retop = True
        return retop

# classe para aglutina os retangulos processados
class SuperRect:
    def __init__(self,rect,idx):
        self.grect = []
        self.grect.append(rect)
        x,y,w,h = rect
        self.x0 = x
        self.y0 = y
        self.w = w
        self.h = h
        self.x1 = x + w
        self.y1 = y + h
        self.idx = idx

    def contains(self,rect):
        ox0,oy0,w,h = rect
        ox1 = ox0 + w
        oy1 = oy0 + h
        if self.x0 > ox1 or ox0 > self.x1:
            return False
        if self.y0 > oy1 or oy0 > self.y1:
            return False
        self.x0 = self.x0 if self.x0 < ox0 else ox0
        self.y0 = self.y0 if self.y0 < oy0 else oy0
        self.x1 = self.x1 if self.x1 > ox1 else ox1
        self.y1 = self.y1 if self.y1 > oy1 else oy1
        self.w = self.x1 - self.x0
        self.h = self.y1 - self.y0
        self.grect.append(rect)
        return True

    def getRect(self):
        return self.x0,self.y0,self.w,self.h

    # ajustar os retangulos aos limites minimos e maximos e retorna os pontos (x0,y0) (x1,y1)
    def getAjRect(self,w,h):
        self.x0 = self.x0 if self.x0 > 0 else 0
        self.y0 = self.y0 if self.y0 > 0 else 0
        self.x1 = self.x1 if self.x1 < w else w
        self.y1 = self.y1 if self.y1 < h else h
        #self.w = self.x1 - self.x0
        #self.h = self.y1 - self.y0
        return (self.x0,self.y0,self.x1,self.y1)

#funcao para aglutinar os subretangulos contidos em listc e retorna a lista se SuperRect aglutinadas
def aglutina(listc):
    lists = []
    listarem = []
    conta = 0
    #ordena os retangulos da maior area para menor area
    listc = sorted(listc,key = lambda re: -re[2]*re[3])
    for riten in listc:
        if riten in listarem:
            continue
        temr = False
        for srit in lists:
            if srit.contains(riten):
                listarem.append(riten)
                temr = True
        if not temr:
            lists.append(SuperRect(riten,conta))
            conta += 1
            listarem.append(riten)
    return lists

#retorna a lista de retangulos aglutinados em duas etapas
def detectMov(img):
    ret,thresh = cv2.threshold(img,127,255,0)
    
    if opencvver[0] == '3' and opencvver[1] == '1':
        #para operar no opencv 3.1
        im2,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    else:
        #para operar no opencv 2.4
        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)<1 :
       return []
   #[(0,0,1,1)],[SuperRect((0,0,1,1),-1)]
    listc = []
    for cnt in  contours:
        x,y,w,h=cv2.boundingRect(cnt)
        px = w/2.5
        py = h/2.5
        x =  int(x - px)
        y =  int(y - py)
        w += int(2*px)
        h += int(2*py)
        if w*h>25:
            listc.append((x,y,w,h))
    lists = aglutina(listc)
    listc2 = []
    for sr in lists:
        listc2.append(sr.getRect())
    lists2 = aglutina(listc2)
    #area de restricao dos retangulos
    (hr, wr) = img.shape[:2]
    rectpts = []
    for reci in lists2:
        rectpts.append(reci.getAjRect(wr,hr))

    return rectpts


if __name__=="__main__":
    print "Carregando video em {}".format(args.video)
    cascade = cv2.CascadeClassifier(args.cascade)
    # o valor e 15 para 30 fps
    # aumentado 50%
    maxttl= int(23.0*float(args.camfps)/30.0)
    # objeto para controle de rastreamento de faces
    trackingF = TrackingFaces(maxttl=maxttl)
    cap = cv2.VideoCapture(args.video)
    temframe=True
    conta = 0
    fatorg=0.33
    #detector de moviemnto implementado no opencv 3.1
    if opencvver[0] == '3' and opencvver[1] == '1':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            fgbg   = cv2.createBackgroundSubtractorKNN()
    else:
            fgbg = cv2.BackgroundSubtractorMOG()
    while(temframe):
            t = clock()
            temframe, frame = cap.read()
            t2 = clock()
            if temframe:
                if rotateIm:
                    # teste com frame rotacionado
                    (h, w) = frame.shape[:2]
                    Mtr = np.float32([[0,1,0],[-1,0,w]])
                    frame = cv2.warpAffine(frame, Mtr, (h,w))

                anotado=frame.copy()
                original = frame.copy()
                #reducao do tamanho pela metade
                frame= cv2.resize(frame,(0,0),fx=fatorg,fy=fatorg)

                conta += 1
                #Obtem as regioes de movimento na imagem
                #detector de moviemnto implementado no opencv 3.1
                if opencvver[0] == '3' and opencvver[1] == '1':
                    fgmask = fgbg.apply(frame)
                    fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel)
                else:
                    fgmask = fgbg.apply(frame,learningRate=0.001)

                if args.displayOn:
                    cv2.imshow("Mask",fgmask)
                rectpts = detectMov(fgmask)
                for x0,y0,x1,y1 in rectpts:
                    x0 = int(x0/fatorg)
                    y0 = int(y0/fatorg)
                    x1 = int(x1/fatorg)
                    y1 = int(y1/fatorg)
                    cv2.rectangle(anotado,(x0,y0),(x1,y1),(0,255,0),3)
                #print "Frame {}".format(conta)
                rects,grayimg =detect(frame, cascade,rectpts)


                listaDeRotulos=trackingF.updateFromRectWH(conta, rects,grayimg)


                cv2.putText(anotado,'F:{}'.format(conta),(0,(20)),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(255,255,255),1)
                if len(rects)>0:
                    drects = []
                    novalistaR = {}
                    for r in rects:
                        x1,y1,w,h=r
                        ides=listaDeRotulos[(x1,y1,w,h)]
                        x1,y1,w,h=(np.array(r)/fatorg).astype(int)
                        drects.append((x1,y1,w,h))
                        novalistaR[(x1,y1,w,h)]=[]
                        novalistaR[(x1,y1,w,h)]=ides
                    draw_rects(anotado,drects,(0,0,255),listaDeRotulos=novalistaR)


                rectst,nomest=trackingF.getOnTrackFaces(conta)

                if len(rectst)>0:
                    drectst = []
                    nnomes = {}
                    for r in rectst:
                        ide = nomest[r]
                        x1,y1,w,h  = (np.array(r)/fatorg).astype(int)
                        drectst.append((x1,y1,w,h))
                        nnomes[(x1,y1,w,h)]=[]
                        nnomes[(x1,y1,w,h)].append(ide)
                    draw_rects(anotado,drectst,(0,255,255),listaDeRotulos=nnomes)

                    #cv2.waitKey(0)
                anotar = cv2.resize(anotado,(0,0),fx=0.5,fy=0.5)
                if args.displayOn:
                    cv2.imshow("Anotado",anotar)
                #Coleta amostras
                if trackingF.collectCurrentSamp(conta,original,fatorg,mdb=mdb,base="corredor"):
                    dt = clock() - t
                    dt2 = clock() - t2
                    print "Total {} ms e {} ms".format(dt*1000,dt2*1000)
                # apresentado amostragem
                #trackingF.drawSampleTrack()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    # When everything done, release the capture
    cap.release()

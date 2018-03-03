#!/usr/bin/env python2
# Autor: Roberto Kenji Hiramatsu
#
# Copyright 2016 Universidade de Sao Paulo
#
# Script para analisar em stream de video tecnicas de detecao e ratreamento em video entrelacado da camara dos deputados com agromeracao
#
#
# Incial Data: 2016-08-30
#              2016-11-01 - acrescentando comentarios para analisar o coletor de dados
#


import numpy as np
import cv2
import cv2.cv as cv
from imutils import paths
import argparse
from common import clock, draw_str
import os,sys
import RepUtil
from RepUtil import Face
import openface
import math

import libCamDepDB.MongoConnCamDep as MongoConnCamDep

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str,
                    help="diretorio contendo video",
                    default='/home/yakumo/HBPVR')
parser.add_argument('--cascade', type=str,
                    help="cascade haar detector",
                    default='../haarcascades/haarcascade_frontalface_alt.xml')

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join('..', 'models')
openfaceModelDir = os.path.join(modelDir, 'openface')

dlibModelDir = os.path.join(modelDir, 'dlib')
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    #default=os.path.join(openfaceModelDir,'treinado-jun16.t7'))
                    default=os.path.join(openfaceModelDir,'nn4.small2.v1.t7'))
#'treinado-jun16.t7'
#parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
#                    default=os.path.join(openfaceModelDir, 'custo9.3.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--cuda', type=bool, default=False)



#crianto instancia para interagir com armazenamento na cloud usp
mdb = MongoConnCamDep()


def detect(img, cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rects = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=1,
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

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def confimaFace(vis_roi):
    rgbFrame = vis_roi.copy()
    bb = align.getLargestFaceBoundingBox(rgbFrame)
    if bb is not None:
        return bb,True
    return (0,0,0,0),False

# extrai representacao e outras informacoes da imagem dada
def extraiMetaFace(imag,align,net):
    [he, wi,p] = imag.shape
    rgbFrame = np.zeros((he, wi, 3), dtype=np.uint8)
    rgbFrame[:, :, 0] = imag[:, :, 0]
    rgbFrame[:, :, 1] = imag[:, :, 1]
    rgbFrame[:, :, 2] = imag[:, :, 2]
    bb = align.getLargestFaceBoundingBox(rgbFrame)
    bbs = [bb] if bb is not None else []
    for bb in bbs:
        landmarks = align.findLandmarks(rgbFrame, bb)
        alignedFace = align.align(args.imgDim, rgbFrame, bb,
                                      landmarks=landmarks,
                                      landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            continue
        represe = net.forward(alignedFace)
        angcab,angvcab,pp = RepUtil.calcHVAngRosto(landmarks[0],landmarks[16],landmarks[27])
        idxB = RepUtil.calcBlur(alignedFace)
        return True,alignedFace,Face(rep=represe, identity=-1,angh = angcab,angv=angvcab ,bluridx=idxB)
    return False,[],None

def desenhaPFid(recorte,img,bb,x0,y0):
    landmarks = align.findLandmarks(recorte, bb)
    reff = [0,8,16,27,30,33,36,45,48,51,54,57,62,66]
    for idxp in reff:
        x,y = landmarks[idxp]
        xc = x0+x
        yc = y0+y
        cv2.circle(img,center=(xc,yc),radius=3,
                               color=(255,204,102),thickness=-1)
    return landmarks

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

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
            soma += RepUtil.calcDisDP((a,b),(c,d))
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

# classe para armazenar faces de um possivel sujeito
class Sujeito:
    def __init__(self,identity):
        # numeracao que relaciona possivel identidade
        self.identity = identity
        # lista de objetos do tipo classe Face
        self.listaFaces = []



    def __repr__(self):
        msgFaces=" "
        for face in self.listaFaces:
            msgFaces = "{}\n{}".format(msgFaces,face)
        return "{{id:{}\nFaces\n{}}}".format(self.identity,msgFaces)

    # metodo para adicionar face ao sujeito
    # tdis - criterio de distancia euclidiana da representacaa
    # tang - valor relativo com relacao ao angulo do rosto na horizontal e vertical sendo o valor default 0.2 igual a 9 graus no calculo relativo
    # tdis com 0.15 e tang 0.25 deixou passar
    # tdis com 0.125 e tang 0.2 deixa passar
    # tdis com 0.1 e tang 0.2 deixa passar pouco em 45 f
    # 0.09 e 0.15 - restritivo? em dois patamares
    # tdis 0.1 e tang 0.17 gera erro de dados
    #considerando a rede nn4.smal2.v1 a distancia de referencia 0.09 da acerto presumido de 99,68%
    #o desvio do rosto de 0.15 representa 6.75 graus de uma imagem para outra
    def adicionaSeProx(self,face,tdis=0.09,tang=0.15):
        tangse = 1.5*tang
        tdisse = tdis/2.0
        for internof in self.listaFaces:
            distancia = RepUtil.calcDistancia(face.rep, internof.rep)
            diffangh = abs(face.angh - internof.angh)
            diffangv = abs(face.angv - internof.angv)
            #print "Subject {} vs face d:{:5.2f} h:{:5.2f} v:{:5.2f}".format(internof.identity,distancia,diffangh,diffangv)
            if (diffangh < tang and diffangv <tang and distancia <tdis) or (diffangh < tangse and diffangv < tangse and distancia < tdisse ):
                self.listaFaces.append(face)
                return True
        return False

    # retorna quantidade de representacoes
    def getQuantRep(self):
        return len(self.listaFaces)

    # metodo para limitar a quantidade de faces
    def otimizaListaFaces(self,quantidade):
        passo = 0.01


def processaSujeitos(corIdentity,listaSujeitos,face):
    identityMat = -1
    for suje in listaSujeitos:
        if suje.adicionaSeProx(face):
            face.identity=suje.identity
            identityMat = suje.identity
            return True,identityMat
    novoS = Sujeito(corIdentity)
    face.identity=corIdentity
    novoS.listaFaces.append(face)
    listaSujeitos.append(novoS)
    return False,identityMat

def revisaListaSujeitos(quantiMin=20,quantiMax=200):
    novalista = []
    # elimina sujeitos que tem menos de quantMin representacoes
    for suje in listaSujeitos:
        if suje.getQuantRep() > quantiMin:
            # se maior que quantidade maxima entao otimiza para ter menos que a quantidade maxia de representates
            if suje.getQuantRep()>quantiMax:
                suje.otimizaListaFaces()
            novalista.append(suje)



if __name__=="__main__":
    jsondb = open("meta.json",'a')
    maxBlurIdx = 200.0
    minBurIdx = 100.0
    args = parser.parse_args()
    net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                              cuda=args.cuda)
    print args.dir
    cascade = cv2.CascadeClassifier(args.cascade)
    print "Carregando lista de arquivos de videos em {}".format(args.dir)
    arquivos = os.listdir(args.dir)
    drefse = 50.0
    align = openface.AlignDlib(args.dlibFacePredictor)
    # numero maximo de quadros ate que este seja descartado
    maxTTL = 5
    # distancia minima entre quadros de localizacao de rosto
    minEntreQ = 100
    contaFrame = 0
    contaArq   = 0
    identity   = 0
    listaSujeitos = []
    for arq in arquivos:
        print "Processando arquivo {}".format(arq)
        contaArq += 1
        #if contaArq < 4:
        #    continue
        cap = cv2.VideoCapture(args.dir+'/'+arq)
        print cap
        conta = 0
        enquadraa = []
        temframe=True
        while(temframe):
            t = clock()
            # Capture frame-by-frame
            temframe, frame = cap.read()
            contaFrame += 1
            if temframe:
                #titulo = 'frame_{:04d}'.format(conta)
                # detecta face no frame
                rects,bdisp=detect(frame, cascade)
                [heb,wib,pb] = frame.shape
                anotado = frame.copy()
                #print heb,wib
                conta = 0
                contad = 0

                # para evitar multiplas localizacos do rosto geradas pelo haar
                listac = []


                disff = 100.0
                #vetor para tratar localizacao do rosto a posteriore
                enquadra = []
                for x1,y1,x2,y2 in rects:
                    #cria janelamento para recorte da imagem e processamento para etapa de dlib
                    x1,y1,x2,y2 = RepUtil.novoEquad(x1,y1,x2,y2,wib,heb)
                    dt = clock() - t
                    print 'D Time',dt,' s'
                    vis_roi = frame[y1:y2, x1:x2]
                    draw_rects(anotado, [(x1,y1,x2,y2)], (0,0,255))
                    #determina se imagen e detectada no dlib
                    bb,ret = confimaFace(vis_roi)
                    if ret:
                        # processa se foi encontrado rosto
                        # desenha os pontos ficuciais
                        newrect = (bb.left(),bb.top(),bb.right(), bb.bottom())
                        # armazena para proxima interacao
                        enquadra.append((x1,y1,x2,y2,maxTTL))
                        xn1,yn1,xn2,yn2 = newrect
                        #print xn1,yn1,xn2,yn2 , x1,y1,x2,y2, (x2-x1),(y2-y1)
                        xp1 = x1 + xn1
                        yp1 = y1 + yn1
                        xp2 = x1 + xn2
                        yp2 = y1 + yn2
                        xc = (xp1+xp2)/2
                        yc = (yp1+yp2)/2
                        p = 0
                        for p1 in listac:
                            disff = RepUtil.calcDisDP((xc,yc), p1)
                            if disff < 25.0:
                                print "Contad:",contad," p: ",p," dis: ",disff," -> ",(xc,yc), p1
                                break
                            p += 1
                        listac.append((xc,yc))
                        #print xp1,yp1,xp2,yp2
                        if disff > 25.0:
                            landmarks=desenhaPFid(vis_roi.copy(),anotado,bb,x1,y1)
                            #gerando angulo normalizado em 45 graus
                            angcab,angvcab,pp = RepUtil.calcHVAngRosto(landmarks[0],landmarks[16],landmarks[27])
                            alignedFace = align.align(args.imgDim, vis_roi.copy(), bb,
                                      landmarks=landmarks,
                                      landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                            idxB = RepUtil.calcBlur(alignedFace)
                            print "Blur idx:",idxB
                            if idxB > maxBlurIdx:
                                #processa desentlelacamento da imagen de interesse
                                resultado,ret=separaOddEven(vis_roi)
                                larguraimg=(bb.right()-bb.left())
                                if ret and larguraimg>108:
                                    print "Largura da imagem: {}".format(larguraimg)
                                    #cv2.imshow("Refrencia",vis_roi)
                                    #cv2.imshow("Aceito_d_disp",resultado)
                                    #cv2.waitKey(0)
                                    ret,alinhado,face = extraiMetaFace(resultado,align,net)
                                    if ret:
                                        retps,identidade = processaSujeitos(identity,listaSujeitos,face)
                                        if not retps:
                                            identity += 1
                                        face.arquivo = "camara_v{:02d}_i{:04d}_f{:05d}.jpg".format(contaArq,face.identity,contaFrame)
                                        cv2.imwrite(face.arquivo,resultado)
                                        infoj = RepUtil.geraJSON(str(contaFrame),str(face.identity),face.angh,face.angv,face.bluridx,face.arquivo,0,netrep=face.rep)
                                        RepUtil.gravaJSON(jsondb,infoj)
                                        cv2.putText(anotado,'I:{}'.format(identidade),(xp1,(yp1+80)),cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.6,(255,255,255),1)
                                        #print face
                                        #print listaSujeitos

                                #else:
                                    #cv2.imshow("rejeitado",resultado)
                                    #cv2.waitKey(0)

                            else:
                                if idxB > minBurIdx:
                                    #cv2.imshow("Aceito_direto",vis_roi)
                                    ret,alinhado,face = extraiMetaFace(vis_roi,align,net)
                                    if ret:
                                        retps,identidade = processaSujeitos(identity,listaSujeitos,face)
                                        if not retps:
                                            identity += 1
                                        face.arquivo = "camara_v{:02d}_i{:04d}_f{:05d}.jpg".format(contaArq,face.identity,contaFrame)
                                        cv2.imwrite(face.arquivo,vis_roi)
                                        infoj = RepUtil.geraJSON(str(contaFrame),str(face.identity),face.angh,face.angv,face.bluridx,face.arquivo,0,netrep=face.rep)
                                        RepUtil.gravaJSON(jsondb,infoj)
                                        cv2.putText(anotado,'I:{}'.format(identidade),(xp1,(yp1+80)),cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.6,(255,255,255),1)
                                        #print face
                                        #print listaSujeitos
                                    #cv2.waitKey(0)
                                #else:
                                    #cv2.imshow("rejeitado_baixo_q",vis_roi)
                                    #cv2.waitKey(0)

                                #cv2.waitKey(0)

                            draw_rects(anotado, [(xp1,yp1,xp2,yp2)], (0,255,0))
                            angcab = angcab*45
                            cv2.putText(anotado,'B:{:5.1f}'.format(idxB),(xp1,(yp1+20)),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(255,255,255),1)
                            cv2.putText(anotado,'H:{:3.0f}'.format(angcab),(xp1,(yp1+40)),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(255,255,255),1)
                            angvcab = angvcab*45
                            cv2.putText(anotado,'V:{:3.0f}'.format(angvcab),(xp1,(yp1+60)),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(255,255,255),1)
                            #if idxB > 500.0:
                            cv2.imshow('video',anotado)
                            cv2.waitKey(0)
                            print "{:2d} {:2d} {:3d}:{:3d}:{:3d}:{:3d} {:3d}:{:3d}:{:3d}:{:3d} {:3d}:{:3d}:{:3d}:{:3d} {:3d}:{:3d} {:3d}:{:3d}".format(
                                conta, contad ,
                                x1,y1,x2,y2,
                                xn1,yn1,xn2,yn2,
                                xp1,yp1,xp2,yp2,
                                xc,yc,
                                (xn2-xn1),(yn2-yn1))
                        contad += 1
                        dt = clock() - t
                        print 'R Time',dt,' s'
                    conta += 1

                dissemin = 100.0
                for x1,y1,x2,y2,ttl in enquadraa:
                    #print "Procesando enquadramento em ",x1,y1,x2,y2
                    vis_roi = frame[y1:y2, x1:x2]
                    draw_rects(anotado, [(x1,y1,x2,y2)], (255/(ttl+2),255,255))
                    bb,ret = confimaFace(vis_roi)
                    if ret:
                        # processa se foi encontrado rosto
                        # desenha os pontos ficuciais
                        newrect = (bb.left(),bb.top(),bb.right(), bb.bottom())
                        # armazena para proxima interacao
                        xn1,yn1,xn2,yn2 = newrect
                        #print xn1,yn1,xn2,yn2 , x1,y1,x2,y2, (x2-x1),(y2-y1)
                        xp1 = x1 + xn1
                        yp1 = y1 + yn1
                        xp2 = x1 + xn2
                        yp2 = y1 + yn2
                        xc = (xp1+xp2)/2
                        yc = (yp1+yp2)/2
                        p = 0
                        for p1 in listac:
                            disffse = RepUtil.calcDisDP((xc,yc), p1)
                            if disffse < drefse:
                                #print "Segundo equadramento Contad:",contad," p: ",p," dis: ",disffse," -> ",(xc,yc), p1
                                break
                            else:
                                if dissemin > disffse:
                                    dissemin = disffse
                            p += 1
                        #print xp1,yp1,xp2,yp2
                        if disffse > drefse:
                            # rosto encontrado em quadro nao detectado no haar cascade
                            enquadra.append((x1,y1,x2,y2,maxTTL))
                            #print  "Segundo equandramento maior que ",drefse," dis min: ",dissemin," para ",(xc,yc)
                            landmarks=desenhaPFid(vis_roi.copy(),anotado,bb,x1,y1)
                            angcab,angvcab,pp = RepUtil.calcHVAngRosto(landmarks[0],landmarks[16],landmarks[27])
                            alignedFace = align.align(args.imgDim, vis_roi.copy(), bb,
                                      landmarks=landmarks,
                                      landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                            idxB = RepUtil.calcBlur(alignedFace)
                            print "Blur idx:",idxB
                            if idxB > maxBlurIdx:
                                resultado,ret=separaOddEven(vis_roi)
                                if ret:
                                    #cv2.imshow("Refrencia",vis_roi)
                                    #cv2.imshow("Aceito_d_disp",resultado)
                                    ret,alinhado,face = extraiMetaFace(resultado,align,net)
                                    if ret:
                                        retps,identidade = processaSujeitos(identity,listaSujeitos,face)
                                        if not retps:
                                            identity += 1
                                        face.arquivo = "camara_v{:02d}_i{:04d}_f{:05d}.jpg".format(contaArq,face.identity,contaFrame)
                                        cv2.imwrite(face.arquivo,resultado)
                                        infoj = RepUtil.geraJSON(str(contaFrame),str(face.identity),face.angh,face.angv,face.bluridx,face.arquivo,0,netrep=face.rep)
                                        RepUtil.gravaJSON(jsondb,infoj)
                                        cv2.putText(anotado,'I:{}'.format(identidade),(xp1,(yp1+80)),cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.6,(255,255,255),1)
                                        #print face
                                        #print listaSujeitos
                                    #cv2.waitKey(0)
                                #else:
                                    #cv2.imshow("rejeitado",resultado)
                                    #cv2.waitKey(0)
                            else:
                                if idxB > minBurIdx:
                                    #cv2.imshow("Aceito_direto",vis_roi)
                                    ret,alinhado,face = extraiMetaFace(vis_roi,align,net)
                                    if ret:
                                        retps,identidade = processaSujeitos(identity,listaSujeitos,face)
                                        if not retps:
                                            identity += 1
                                        face.arquivo = "camara_v{:02d}_i{:04d}_f{:05d}.jpg".format(contaArq,face.identity,contaFrame)
                                        cv2.imwrite(face.arquivo,vis_roi)
                                        infoj = RepUtil.geraJSON(str(contaFrame),str(face.identity),face.angh,face.angv,face.bluridx,face.arquivo,0,netrep=face.rep)
                                        RepUtil.gravaJSON(jsondb,infoj)
                                        cv2.putText(anotado,'I:{}'.format(identidade),(xp1,(yp1+80)),cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.6,(255,255,255),1)
                                        #print face
                                        #print listaSujeitos
                                    #cv2.waitKey(0)
                                #else:
                                    #cv2.imshow("rejeitado_baixo_q",vis_roi)
                                    #cv2.waitKey(0)
                            angcab = angcab*45
                            draw_rects(anotado, [(xp1,yp1,xp2,yp2)], (255,0,0))
                            cv2.putText(anotado,'B:{:5.1f}'.format(idxB),(xp1,(yp1+20)),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(255,255,255),1)
                            cv2.putText(anotado,'H:{:3.0f}'.format(angcab),(xp1,(yp1+40)),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(255,255,255),1)
                            angvcab = angvcab*45
                            cv2.putText(anotado,'V:{:3.0f}'.format(angvcab),(xp1,(yp1+60)),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(255,255,255),1)
                            print "Segundo :{:2d} {:2d} {:3d}:{:3d}:{:3d}:{:3d} {:3d}:{:3d}:{:3d}:{:3d} {:3d}:{:3d}:{:3d}:{:3d} {:3d}:{:3d} {:3d}:{:3d} ttl:{}".format(
                                conta, contad ,
                                x1,y1,x2,y2,
                                xn1,yn1,xn2,yn2,
                                xp1,yp1,xp2,yp2,
                                xc,yc,
                                (xn2-xn1),(yn2-yn1),ttl)
                            cv2.imshow('video',anotado)
                            cv2.waitKey(0)
                        contad += 1
                        dt = clock() - t
                        print 'SE R Time',dt,' s'
                    else:
                        #print "Interacao de quandro sen detecao em ",x1,y1,x2,y2," com ttl :",ttl
                        if ttl > 0:
                            distqref = 2*minEntreQ
                            #processa se ttl ainda esta ativo
                            for xe1,ye1,xe2,ye2,ttle in enquadra:
                                # determina o centro do quadro armazenado corrente
                                xce = (xe2-xe1)/2
                                yce = (ye2-ye1)/2
                                xc  = (x2 - x1)/2
                                yc  = (y2 - y1)/2
                                distq = RepUtil.calcDisDP((xc,yc), (xce,yce))
                                if distqref > distq:
                                    distqref = distq
                            if distqref > minEntreQ :
                                print "Quadro nao detectado para proxima iteracao em ",x1,y1,x2,y2," com ttl :",(ttl-1)
                                enquadra.append((x1,y1,x2,y2,ttl-1))
                                #cv2.imshow('video',anotado)
                                #cv2.waitKey(0)

                    conta += 1
                enquadraa = enquadra
                print "Proximo teste de equadramento com ",len(enquadraa)
                #cv2.imshow('video',anotado)
                #if disff < 25.0:
                #    if cv2.waitKey(5000) & 0xFF == ord('q'):
                #        break
                #else:
                dt = clock() - t
                print 'Total Time',dt,' s'
                #if cv2.waitKey(10) & 0xFF == ord('q'):
                #    break
                #cv2.destroyWindow(titulo)
            conta += 1
        cap.release()

#!/usr/bin/env python2
# Autor: Roberto Kenji Hiramatsu
#
# Copyright 2016 Universidade de Sao Paulo
#
# Tratador para processamento de representacoes e funcoes relacionadas
# Data inicial: 2016-08-24
# 2016-08-26: Mudanca da classe Face para incluir informacoes sobre
#             posicionamento angular horizonal angh e vertical angv
#             da cabeca com meia volta de circulo representado 4

from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC
import cv2
import cv2.cv as cv
import openface
import math
from scipy.optimize import fsolve
import json

from math import  acos,atan,degrees
#from boto.gs import resumable_upload_handler

class FaceA:

    def __init__(self, rep, arquivo):
        self.rep = rep
        self.arquivo = arquivo
        self.selec = True


    def __repr__(self):
        return "{{id: {}, rep[0:5]: {}}}".format(
            self.arquivo,
            self.rep[0:5]
        )

class TriAng:
    def __init__(self,ang1,ang2,ang3):
        self.ang1 = ang1
        self.ang2 = ang2
        self.ang3 = ang3

    def __repr__(self):
        return "{{a1:{:5.1f}, a2:{:5.1f}, a3:{:5.1f}}}".format(
            self.ang1,self.ang2,
            self.ang3
        )

def calcDistancia(r1,r2):
    soma = 0.0
    for idx, val in enumerate(r1):
        pa=r1[idx]-r2[idx]
        soma += pa * pa
    return soma

class Face:

    def __init__(self, rep, identity,arquivo = None,triangs = None, angh = 0,angv=0 ,bluridx=0.0):
        self.rep = rep
        self.identity = identity
        self.arquivo = arquivo
        self.triangs = triangs
        self.angh = angh
        self.angv = angv
        self.bluridx = bluridx

    def __repr__(self):
        return "{{id: {},arq: {},ah:{:4.2f},av:{:4.2f},rep[0:5]: {}, {},{:5.1f}}}".format(
            str(self.identity),self.arquivo,self.angh,self.angv,
            self.rep[0:5],self.triangs,self.bluridx
        )
    def compara(self,outro):
        dista = calcDistancia(self.rep, outro.rep)
        da1 =self.triangs.ang1-outro.triangs.ang1
        da2 =self.triangs.ang2-outro.triangs.ang2
        da3 =self.triangs.ang3-outro.triangs.ang3
        #return (dist,TriAng((a1-oa1),(a2-oa2),(a3-oa3)))
        print "{:6.3f} ; {:6.1f} ; {:6.1f} ; {:6.1f} ; {:6.1f} ; {:6.1f} ; {} ; {} ; {} ; {} ; {}".format(
              dista,da1,da2,da3,self.bluridx,outro.bluridx,
              (self.posf > 0 ),(outro.posf > 0),
              self.identity,outro.identity,(self.identity == outro.identity))

def reduzGrupo(grupo):
    maxsubgrupo = []
    maxele = 0
    for ix in range(0,len(grupo)):
        subgrupo = []
        subgrupo.append(grupo[ix])
        for jx in range ((ix+1),len(grupo)):
            if calcDistancia(grupo[ix],grupo[jx]) < 0.8 :
                subgrupo.append(grupo[jx])
        if len(subgrupo) > maxele:
            maxele = len(subgrupo)
            maxsubgrupo = subgrupo
    return maxsubgrupo


def mediaRep(grupo):
    resp = [ 0 for x in range(len(grupo[0]))]
    n = 0
    for rep in grupo:
        for idx , val in enumerate(rep):
            resp[idx] += rep[idx]
        n += 1
    for idx , val in enumerate(resp):
        resp[idx] /= n
    #print n,resp
    return resp

# separar os arquivos que apresentam distancia menor que valor de corte
# com valor default de 0.1
def separaGrupos(listaAR,corte=0.1):
    listaGrupo = []
    for ix in range(0,len(listaAR)):
        subgrupo = []
        subgrupo.append(listaAR[ix])
        for jx in range ((ix+1),len(listaAR)):
            if (calcDistancia(listaAR[ix].rep,listaAR[jx].rep) < corte ) \
            and listaAR[ix].selec :
                listaAR[ix].selec = False
                subgrupo.append(listaAR[jx])
        listaGrupo.append(subgrupo)
    return listaGrupo

def trainSVM(grupos,people):
    d = getData(grupos)
    if d is None:
        svm = None
        return
    else:
        (X, y) = d
        numIdentities = len(set(y + [-1]))
        if numIdentities <= 1:
            return

        param_grid = [
                {'C': [1, 10, 100, 1000],
                 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000],
                 'gamma': [0.001, 0.0001],
                 'kernel': ['rbf']}
            ]
        svm = GridSearchCV(SVC(C=1), param_grid, cv=5).fit(X, y)
    return svm

def mediaDeGrupo(images,people):
    media = []
    i = 0
    grupos = []
    for indv in people:
        grupo = []
        for img in images.values():
            if i == img.identity:
                grupo.append(img.rep)
        print people[i],' de ',len(grupo),' para '
        grupo = reduzGrupo(grupo)
        grupos.append(grupo)
        print len(grupo)
        media.append(mediaRep(grupo))
        i = i+1
    svm = trainSVM(grupos,people)
    return (media,svm)

def getData(grupos):
    #para juntar as informacoes a serem treinadas com svm com valores formados por representacao
    X = []
    y = []
    idx = 0
    for grupo in grupos:
        for rep in grupo:
            X.append(rep)
            y.append(idx)
        idx += 1
    return (X,y)

def isBlur(img,corte):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #restringindo area de processamento para 48x72 regiao dos olhos
    snapg = gray[0:48,12:84]
    #fm = cv2.Laplacian(gray,cv2.CV_64F).var()
    fm = cv2.Laplacian(snapg,cv2.CV_64F).var()
    if fm < corte:
        return True;
    else:
        return False

def calcBlur(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #restringindo area de processamento para 48x72 regiao dos olhos
    snapg = gray[0:48,12:84]
    #fm = cv2.Laplacian(gray,cv2.CV_64F).var()
    return cv2.Laplacian(snapg,cv2.CV_64F).var()



    # coleta os pontos que formam o triangulo de posicionamento da face
    # e calcula os angulos dos tringulos
def coletaVert(landmarks):
    vert = []
    for p in openface.AlignDlib.OUTER_EYES_AND_NOSE:
        vert.append(landmarks[p])
    return vert
# Para determinar a posicao do nariz em relacao a cabeca
# quando negativo o nariz esta aa esquerda da cabeca
# quando positivo o nariz esta aa direita da cabeca
def definePosF(vert):
    [[x0,y0],[x1,y1],[x2,y2]]=vert
    mediH = (x1+x0)/2
    return 100.0*float(x2 - mediH)/float(x1-x0)

#calcula o nivel de diferencia dos triangulos retorna na classe TriAng com tres valores
def nivelSimTri(t1,t2):
    return TriAng((t2.ang1-t1.ang1),(t2.ang2-t1.ang2),(t2.ang3-t1.ang3))

# Calcula os angulos internos de um triangulo no plano
# vert - tres pontos no plano
def calcTriangAng(vert):
    [[x0,y0],[x1,y1],[x2,y2]]=vert
    a = float(x1 - x0)
    b = float(y1 - y0)
    c = float(x2 - x0)
    d = float(y2 - y0)
    ang1 =(degrees(atan(d/c))-degrees(atan(b/a)))
    #print a,b,c,d,ang1
    a = float(x0 - x1)
    b = float(y0 - y1)
    c = float(x2 - x1)
    d = float(y2 - y1)
    p1 = degrees(atan(d/c))
    p2 = degrees(atan(b/a))
    ang2 =(p2-p1)
    ang3 = 180.0 - ang1 - ang2
    return TriAng(ang1,ang2,ang3)

#funcao para gerar regiao de recorte valor incial em 1/7 aumentado para 1/6
fatorenq = 6
def novoEquad(x1,y1,x2,y2,wib,heb):
    #x1,y1,x2,y2 sao as posicoes da face obtida pelo opencv
    #wib e heb sao a largura e altura da imagem
    #este algoritmo acrescenta uma pequena margem a imagem

    dx = (x2-x1)/fatorenq
    x1 = (x1 - dx) if (x1 - dx)>0 else 0
    x2 = (x2 + dx) if (x2 + dx)<wib else wib
    dy = (y2 -y1)/fatorenq
    y1 = (y1 - dy) if (y1 - dy)>0 else 0
    y2 = (y2 + 2*dy) if (y2 + 2*dy) < heb else heb
    return x1,y1,x2,y2

def novoEquadR(rect,wib,heb):
    x1,y1,x2,y2 = rect
    dx = (x2-x1)/fatorenq
    x1 = (x1 - dx) if (x1 - dx)>0 else 0
    x2 = (x2 + dx) if (x2 + dx)<wib else wib
    dy = (y2 -y1)/fatorenq
    y1 = (y1 - dy) if (y1 - dy)>0 else 0
    y2 = (y2 + 2*dy) if (y2 + 2*dy) < heb else heb
    return x1,y1,x2,y2

def calcDisDP(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    dx = float(x2-x1)
    dy = float (y2 - y1)
    return math.sqrt(dx*dx+dy*dy)

# funcao que retorna a indexacao do array de angulos dado um angulo de entrada
# funciona tanto para angulos verticais quanto horizontais
# retorna um indice de 0 a 23 (considerando uma precisao de 5 graus)
def getHashByAngle(ang, astep=15,amax=45):
    tmp = ang
    if tmp > amax:
        tmp = amax
    elif tmp < -amax:
        tmp = -amax

    return int((amax-tmp)/astep)

# funcao para obter o angulo posicionamento horizontal de um rosto
# regang valor de referencia (estimado) para angulo de triangulo formado entre as pontas da orelha e centro do rosto
# @return - retorna angulo em radianos
def calcHAngRosto(dre,drd,refang=math.pi/6.0):
    R = drd / dre
    #funcao trigonometrica para estabelecer relacao de projecao horizontal da imagem
    func  = lambda ang : R -((math.cos(refang)*math.cos(ang)-math.sin(refang)*math.sin(ang))/(math.cos(refang)*math.cos(ang)+math.sin(refang)*math.sin(ang)))
    ang_g = math.pi/4.0 if dre > drd else -math.pi/4.0
    ang_s = fsolve(func,ang_g)
    return ang_s

# calcular angulos da cabeca usando 3 pontos fiduciais + reserva da ponta do nariz
# @return - retorna respectivamente angulo normalizado em 45 graus para horizontal e vertical de
# rosto e tambem o valor do ponto projetado xp,yp do centro da cabeca ao fundo
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
        dre = calcDisDP([xp,yp],pre)
        drd = calcDisDP([xp,yp],prd)
        dpp = calcDisDP([xp,yp],prc)
    # angulo de inclinacao horizonal
    angh = calcHAngRosto(dre,drd)
    # distancia normalizada horizontal
    ded = (dre+drd)/math.cos(angh)
    # distancia de plano de profundidade centro
    dpc = ded * math.sin(refang)/2.0
    # angulo de inclinacao vertical de rosto
    try:
        angv = math.asin(dpp/dpc)
    except ValueError:
        angv = -1.0
        print 'Erro de asin ',dpp,dpc
    angv = angv if yp > y3 else -angv

    return (angh[0]*4.0/math.pi),(angv*4.0/math.pi),(int(xp),int(yp))

def generateRep(netrep):
    seria = '['
    for v in netrep:
        seria += str(v)+','
    return seria+'0]'



def geraJSON(timestamp,name,anghcab,angvcab,idxBlur,filename,classifica,netrep):
    info = { 'timestamp': timestamp,
            'name':name,
            'anghori': anghcab,
            'angvert':angvcab,
            'bluridx':idxBlur,
            'filename':filename,
            'isok':classifica,
            'netrep': generateRep(netrep)}
    return info

def gravaJSON(jsondb,info):
    jsondb.write(json.dumps(info,separators=(',',':'))+"\n\r")

def detectFace(img, cascade):
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



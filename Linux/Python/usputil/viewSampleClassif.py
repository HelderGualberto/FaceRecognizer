#!/usr/bin/env python2
# Autor: Roberto Kenji Hiramatsu
#
# Copyright 2016 Universidade de Sao Paulo
#
# Visualizacao de dados da base coletada com anotacao via servidor
#
# Data inicial: 2016-12-01 - inicial
#               2017-03-06 - realocacao dos caminhos
#               2017-03-07 - processamento de modo texto e coleta em 
#                            arquivo dos resultados

import os
import libColetaFaceDB
from bson.son import SON
import cv2
import numpy as np
import argparse
from websocket import create_connection
import json
import socket
import thread
#import RepUtil

basedir=os.path.join('..','..','data')
parser = argparse.ArgumentParser()

#alt2 apresenta melhor desempenho para detecao de face
parser.add_argument('--cascade', type=str,
                    help="cascade haar detector",
                    #default='haarcascades/haarcascade_frontalface_alt_tree.xml')
                    #default='haarcascades/haarcascade_frontalface_alt2.xml')
                    default=os.path.join(basedir,'haarcascades','haarcascade_frontalface_alt2.xml')) #D:\\app\\opencv31\\sources\\data\\
                    #default='haarcascades/haarcascade_frontalface_default.xml')
                    #default='haarcascades/haarcascade_frontalface_alt.xml')
                    #default='D:\\app\\opencv31\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml')

parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(basedir,"models","dlib", "shape_predictor_68_face_landmarks.dat"))

parser.add_argument('--host', type=str, help="Host to mongodb",
                    #default="192.168.10.236")
                    default="orion.pad.lsi.usp.br")

parser.add_argument('--port', type=int, help="Port of  mongodb",
                    default=37027)

parser.add_argument('--hostPose', type=str, help="Host to classification info mongodb",
                    #default="192.168.10.236")
                    default="kurentofront.pad.lsi.usp.br")

parser.add_argument('--portPose', type=int, help="Port of classification info mongodb",
                    default=37027)


parser.add_argument('--base', type=str, help="Database on  mongodb",
                    #default="huaweicam")
                    #default="huaweicam2s")
                    #default="huaweicam3s")
                    #default="huaweicam4s")
                    default="huaweicam13s")


parser.add_argument('--align', type=bool, help="Alinhamento para entrada na rede neural",
                    default=False)

parser.add_argument('--wsrecognition', type=str, help="Servico para fornecer processamento de reconhecimento de face",
                    default="ws://192.168.102.129:8081")

parser.add_argument('--displayResult', type=bool, help="Apresenta graficamente resultado gerado",
                    default=False)

args = parser.parse_args()
mdb = libColetaFaceDB.MongoConn(url="mongodb://{}:{}".format(args.host,args.port),
                                dbs=args.base)



ret = mdb.db.fs.files.aggregate([
                        #{"$match":{"base":args.base}},
                        {"$group":{"_id":{"identity":"$identity"},"count":{"$sum":1}}},
                        {"$sort": SON([("_id", 1)])}
                        ])
#,
#                        {"$sort": SON([("count", -1), ("_id", 1)])}
from align_dlib import AlignDlib
# alocado globalmente o extrador dlib
align = AlignDlib(args.dlibFacePredictor)

ws = None


# abre a conexao para o servidor
try:
    ws = create_connection(args.wsrecognition)
except socket.error:
    print "Can t connect"
    ws = None


def initMongoConn():
    if ws is None:
        return
    # configura o servidor mongodb no servico remoto
    infodb={}
    infodb['type']="MONGOSRVINI"
    infodb['base']=args.base
    infodb['port']=args.port
    infodb['host']=args.host
    ws.send(json.dumps(infodb))
    raw=ws.recv()
    msg = json.loads(raw)
    print msg

def initPoseDb():
    if ws is None:
        return
    # configura o servidor mongodb no servico remoto
    infodb={}
    infodb['type']="MONGOPOSEDB"
    infodb['port']=args.portPose
    infodb['host']=args.hostPose
    ws.send(json.dumps(infodb))
    raw=ws.recv()
    msg = json.loads(raw)
    print msg

def printInfo(txt):
    print txt

def queryId(identity):
    if ws is None:
        return {}
    infodb={}
    infodb['type']="QUERYIDINFO"
    infodb['identity']=identity
    ws.send(json.dumps(infodb))
    inprogress = True
    while inprogress:
        raw=ws.recv()
        msg = json.loads(raw)
        if msg['RESPONSE'] == 'OK' or msg['RESPONSE']=='ERROR':
            inprogress = False
        elif msg['RESPONSE'] == 'INPROGRESS':
            printInfo(msg['INFO'])
    return msg

def calcDisR(a,b):
    subrep = np.subtract(a,b)
    vmrep = np.multiply(subrep,subrep)
    return vmrep.sum()

def calcBlur(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #restringindo area de processamento para 48x72 regiao dos olhos
    snapg = gray[0:48,12:84]
    #fm = cv2.Laplacian(gray,cv2.CV_64F).var()
    return cv2.Laplacian(snapg,cv2.CV_64F).var()

# classe para gerar valor combinado da probabilidade baseado em tres premissas
# resultado gerado da probabilidade e idenpendente
# o valor a qual gera a probalidade e proporcional ao quadradado do tamanho
# da entrada da rede neural com valor limite em 96 (versao nn4.smallv2)
# o resultado leva em consideracao a ponderacao no resultado geral
class PredicProb:
    def __init__(self):
        self.acprob=0
        self.positSum = 0


    #Usando a area para reduzir a influencia da probabilidade
    # necessita de algum modelo mais concreto para relacionar a area a
    # probabilidade gerada
    def updatePositive(self,curwid,prob,ppor):
        self.positSum += ppor
        pprob = self.acprob+prob*ppor-self.acprob*prob*ppor
        self.acprob= pprob



    #Fazendo que a probabilidade seja proporcional ao numero de entradas
    # necessita de modelo tambem para relacionar esta probabilidade
    def getCurProb(self,totalSum):
        return self.acprob*self.positSum/totalSum

refwid = 96.0
contagemIL = 1

def updateContagemIL():
    global contagemIL
    contagemIL += 1

def apresenta(identity):
    sret = mdb.db.fs.files.find({'identity':identity}).sort("seq",1)
    info = queryId(identity)
    largeimg = np.zeros((980,1200,3),np.uint8)
    xpos = 0
    ypos = 0
    lastRep = None
    lastRep2 = None
    lastRep3 = None
    lastRep4 = None
    lastDate = 0
    lastHour = 0
    listaDeNomes={}
    totalSum = 0
    contaMatR = 0
    for dimg in sret:
        img=mdb.readIm(dimg)#leArquivo(dimg)
        #img = cv2.imread(os.path.join("img",dimg["filename"]))
        if img is None:
            print "File {} is empty".format(dimg["filename"])
            continue
        if args.align:
            bb = align.getLargestFaceBoundingBox(img)
            if bb is None:
                print "O arquivo {} nao detectou face".format(dimg["filename"])
                continue
            img = img[bb.top():bb.bottom(),bb.left():bb.right()].copy()
        h,w,c = img.shape
        if h == 0 or w == 0:
                print "Image {} storage with strange format {}x{}".format(dimg["filename"],h,w)
                continue
        # largura da imagem recortada usada para alinhamento da imagem de entrada da rede
        # neural
        wref = w
        ppor = float(wref)/refwid if wref < refwid else 1.0
        # nao exibe se a larugra da imagem form menor que metade da largura de input
        if ppor < 0.75:
            continue

        if w > 120:
            fr = 120.0/float(w)
            #print "Redimensionando devido a {} com fator {}".format(w,fr)
            try:
                 img = cv2.resize(img,(0,0),fx=fr,fy=fr)
            except(cv2.error,'OpenCV Error'):
                 print "Error in resize image {} with shape: {} ".format(dimg["filename"],img.shape)
            h,w,c = img.shape
        #cv2.imshow("Olhos",img[h/5:h*3/5,w/5:w*4/5].copy())
        idxB = calcBlur(img[h/5:h*3/6,w/5:w*4/5].copy())
        # imagem com qualidade muito baixa
        if idxB <30 :
            continue

        ppor = ppor*ppor
        totalSum += ppor


        largeimg[ypos:ypos+h,xpos:xpos+w]=img
        cv2.putText(largeimg,str(dimg["seq"]),(xpos,ypos+40),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),1)

        textowi = "L{:03d} B{:4.0f}".format(wref,idxB)
        cv2.putText(largeimg,textowi,(xpos,ypos+60),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),1)
        #print info.keys()


        if u"angh{:03d}".format(dimg['seq']) in info.keys():
            texto = "h{:2.1f} v{:2.1f}".format(info[u"angh{:03d}".format(dimg['seq'])],
                                                              info[u"angv{:03d}".format(dimg['seq'])])
            #print texto
            cv2.putText(largeimg,texto,
                        (xpos,ypos+80),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),1)
            ra = np.array(info[u"rep{:03d}".format(dimg['seq'])])
            if lastRep is not None:
                valor = calcDisR(ra,lastRep)
                texto = "{:3.2f}".format(valor)
                cv2.putText(largeimg,str(texto),
                        (xpos,ypos+100),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),1)
                if lastRep2 is not None:
                    valor = calcDisR(ra,lastRep2)
                    texto = "{:3.2f}".format(valor)
                    cv2.putText(largeimg,str(texto),
                        (xpos+60,ypos+100),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),1)
                if lastRep3 is not None:
                    valor = calcDisR(ra,lastRep3)
                    texto = "{:3.2f}".format(valor)
                    #cv2.putText(largeimg,str(texto),
                    #    (xpos,ypos+120),cv2.FONT_HERSHEY_SIMPLEX,
                    #            0.6,(0,255,255),1)

                if lastRep4 is not None:
                    valor = calcDisR(ra,lastRep4)
                    texto = "{:3.2f}".format(valor)
                    #cv2.putText(largeimg,str(texto),
                    #    (xpos+60,ypos+120),cv2.FONT_HERSHEY_SIMPLEX,
                    #            0.6,(0,255,255),1)
            if u"name{:03d}.00".format(dimg['seq']) in info.keys():
                contaMatR += 1
                inforeco = "{} {:4.1f}".format(info[u"name{:03d}.00".format(dimg['seq'])],
                                           info[u"prob{:03d}.00".format(dimg['seq'])]*100)
                if info[u"prob{:03d}.00".format(dimg['seq'])] >0.5:
                    cv2.putText(largeimg,inforeco,
                                (xpos,ypos+120),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,0),1)
                else:
                    cv2.putText(largeimg,inforeco,
                                (xpos,ypos+120),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,0,255),1)
                naoTem = True
                for nome in listaDeNomes.keys():
                    if info[u"name{:03d}.00".format(dimg['seq'])] == nome:
                        listaDeNomes[nome].updatePositive(wref,info[u"prob{:03d}.00".format(dimg['seq'])],ppor)
                        naoTem = False
                if naoTem:
                    nome = info[u"name{:03d}.00".format(dimg['seq'])]
                    listaDeNomes[nome]= PredicProb()
                    listaDeNomes[nome].updatePositive(wref,info[u"prob{:03d}.00".format(dimg['seq'])],ppor)

                # determina a maior probabilidade
                nomeSel ='Unknow'
                prob = 0.0
                for  nome in listaDeNomes.keys():
                    tempp = listaDeNomes[nome].getCurProb(totalSum)
                    if tempp > prob:
                        prob = tempp
                        nomeSel = nome
                inforeco = "{} {:4.1f}".format(nomeSel,
                                           prob*100)
                cv2.putText(largeimg,inforeco,
                                (xpos,ypos+160),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,0),1)



            if u"name{:03d}.01".format(dimg['seq']) in info.keys():
                inforeco = "{} {:4.1f}".format(info[u"name{:03d}.01".format(dimg['seq'])],
                                           info[u"prob{:03d}.01".format(dimg['seq'])]*100)
                if info[u"prob{:03d}.01".format(dimg['seq'])] >0.5:
                    cv2.putText(largeimg,inforeco,
                                (xpos,ypos+140),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,0),1)
                else:
                    cv2.putText(largeimg,inforeco,
                                (xpos,ypos+140),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,0,255),1)






            lastRep4 = lastRep3
            lastRep3 = lastRep2
            lastRep2 = lastRep
            lastRep = ra


        if xpos < 900:
            xpos += 180
        else:
            xpos = 0
            if ypos < 630:
                ypos += 210
            else:
                cv2.putText(largeimg,str(identity),(0,ypos+20),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),3)
                if args.displayResult:
                    cv2.imshow("Listagem",largeimg)
                    if cv2.waitKey(0)& 0xFF == ord('q'):
                        return False
                else:
                    cv2.imwrite("listagem{:04d}.jpg".format(contagemIL),largeimg)
                    updateContagemIL()   
                ypos = 0
                largeimg = np.zeros((980,1200,3),np.uint8)
        lastDate = dimg["yemoda"]
        lastHour = dimg["homi"]
    cv2.putText(largeimg,"{} {:6d} {:4d}".format(identity,lastDate,lastHour),(0,ypos+20),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),3)

    if len(listaDeNomes.keys())>0 and contaMatR>2:
        ypos += 180
        for nome in listaDeNomes.keys():
            inforeco = "{} {:4.1f}".format(nome,
                                           listaDeNomes[nome].getCurProb(totalSum)*100)
            cv2.putText(largeimg,inforeco,
                                (0,ypos),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(255,0,0),1)
            ypos += 20
    if args.displayResult:
        cv2.imshow("Listagem",largeimg)
        if cv2.waitKey(0)& 0xFF == ord('q'):
            return False
    else:
       cv2.imwrite("listagem{:04d}.jpg".format(contagemIL),largeimg)
       updateContagemIL()
    return True


if __name__ == '__main__':
    initMongoConn()
    initPoseDb()

    for doc in ret:
        print "{} {}".format(doc["_id"]["identity"],doc["count"])
        if not apresenta(doc["_id"]["identity"]):
            break
    if ws is not None:
        ws.close()

#!/usr/bin/env python2
# Autor: Roberto Kenji Hiramatsu
#
# Copyright 2016 Universidade de Sao Paulo
#
# Visualizacao de dados da base coletada com anotacao via servidor
#
# Data inicial: 2016-12-01 - inicial

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

parser = argparse.ArgumentParser()

#alt2 apresenta melhor desempenho para detecao de face
parser.add_argument('--cascade', type=str,
                    help="cascade haar detector",
                    #default='haarcascades/haarcascade_frontalface_alt_tree.xml')
                    #default='haarcascades/haarcascade_frontalface_alt2.xml')
                    default='haarcascades\\haarcascade_frontalface_alt2.xml') #D:\\app\\opencv31\\sources\\data\\
                    #default='haarcascades/haarcascade_frontalface_default.xml')
                    #default='haarcascades/haarcascade_frontalface_alt.xml')
                    #default='D:\\app\\opencv31\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml')

parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join("models","dlib", "shape_predictor_68_face_landmarks.dat"))

parser.add_argument('--host', type=str, help="Host to mongodb",
                    #default="192.168.10.236")
                    default="orion.pad.lsi.usp.br")

parser.add_argument('--port', type=int, help="Port of  mongodb",
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


def printInfo(txt):
    print txt

def queryId(identity):
    if ws is None:
        return {}
    infodb={}
    infodb['type']="QUERYIDREP"
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
    lastQuad = []
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
        wref = w
        if w > 120:
            fr = 120.0/float(w)
            #print "Redimensionando devido a {} com fator {}".format(w,fr)
            try:
                 img = cv2.resize(img,(0,0),fx=fr,fy=fr)
            except(cv2.error,'OpenCV Error'):
                 print "Error in resize image {} with shape: {} ".format(dimg["filename"],img.shape)
            h,w,c = img.shape
        largeimg[ypos:ypos+h,xpos:xpos+w]=img
        cv2.putText(largeimg,str(dimg["seq"]),(xpos,ypos+40),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),1)
        cv2.putText(largeimg,str(wref),(xpos,ypos+60),cv2.FONT_HERSHEY_SIMPLEX,
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
            curQuad = []
            if lastRep is not None:
                valor = calcDisR(ra,lastRep)
                curQuad.append(valor)
                texto = "{:3.2f}".format(valor)
                cv2.putText(largeimg,str(texto),
                        (xpos,ypos+100),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),1)
                if lastRep2 is not None:
                    valor = calcDisR(ra,lastRep2)
                    curQuad.append(valor)
                    texto = "{:3.2f}".format(valor)
                    cv2.putText(largeimg,str(texto),
                        (xpos+60,ypos+100),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),1)
                if lastRep3 is not None:
                    valor = calcDisR(ra,lastRep3)
                    curQuad.append(valor)
                    texto = "{:3.2f}".format(valor)
                    cv2.putText(largeimg,str(texto),
                        (xpos,ypos+120),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),1)

                if lastRep4 is not None:
                    valor = calcDisR(ra,lastRep4)
                    curQuad.append(valor)
                    texto = "{:3.2f}".format(valor)
                    cv2.putText(largeimg,str(texto),
                        (xpos+60,ypos+120),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),1)
                    if len(lastQuad)>0 :
                        laq = np.array(lastQuad)
                        mla = np.mean(laq)
                        sla = np.std(laq)
                        lcu = np.array(curQuad)
                        mcu = np.mean(lcu)
                        scu = np.std(lcu)
                        difm = abs(mla-mcu)

                        if (sla+scu)*6 < difm:
                            print "mla:{} sla:{} mcu:{} scu:{} difm:{}".format(mla,sla,mcu,scu,difm)
                            cv2.imshow("Listagem",largeimg)
                            if cv2.waitKey(0)& 0xFF == ord('q'):
                                return False
                    lastQuad = curQuad

            lastRep4 = lastRep3
            lastRep3 = lastRep2
            lastRep2 = lastRep
            lastRep = ra


        if xpos < 1080:
            xpos += 120
        else:
            xpos = 0
            if ypos < 700:
                ypos += 120
            else:
                cv2.putText(largeimg,str(identity),(0,ypos+20),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),3)
                cv2.imshow("Listagem",largeimg)
                if cv2.waitKey(1)& 0xFF == ord('q'):
                    return False
                ypos = 0
                largeimg = np.zeros((980,1200,3),np.uint8)
        lastDate = dimg["yemoda"]
        lastHour = dimg["homi"]
    cv2.putText(largeimg,"{} {:6d} {:4d}".format(identity,lastDate,lastHour),(0,ypos+20),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),3)
    cv2.imshow("Listagem",largeimg)
    if cv2.waitKey(0)& 0xFF == ord('q'):
        return False
    return True


if __name__ == '__main__':
    initMongoConn()
    for doc in ret:
        print "{} {}".format(doc["_id"]["identity"],doc["count"])
        if not apresenta(doc["_id"]["identity"]):
            break
    if ws is not None:
        ws.close()

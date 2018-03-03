#!/usr/bin/env python2
# Autor: Roberto Kenji Hiramatsu
#
# Copyright 2016 Universidade de Sao Paulo
#
# Analise de Nearest Neighbors
#
# Data inicial: 2016-12-06 - inicial

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
#from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

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
        #elif msg['RESPONSE'] == 'INPROGRESS':
        #    printInfo(msg['INFO'])
    return msg

def calcDisR(a,b):
    subrep = np.subtract(a,b)
    vmrep = np.multiply(subrep,subrep)
    return vmrep.sum()

#analisa se elementos da classe sao conpativeis
def anlyseCluster(cluster,reglogit,trh=0.5,nc=10):
    listad = []
    contan=0
    for idx1,c1 in enumerate(cluster):
        for idx2,c2 in enumerate(cluster):
            if idx1 <= idx2:
                continue
            disc = calcDisR(c1, c2)
            pb=logreg.predict_proba(np.array([disc]).reshape(1,-1))
            #print "disc: {} pb:{}".format(disc,pb)
            if pb[0,1] < trh:
                listad.append(idx1)
                listad.append(idx2)
            else:
                contan += 1
    h,e = np.histogram(listad,range(0,nc+2))
    #print "l{} h {} e{} n{}".format(listad,h,e,contan)
    idxr = -1
    max =  1
    for idx,v in enumerate(h):
        if max < v:
            max = v
            idxr = idx
    if max > 2:
        print "Idxr {} max= {}".format(idxr,max)
        return True,idxr
    return False,-1

def reformKmeans(kmeans,reglogit,lis,identity,nc=10):
    ret,idxbad = anlyseCluster(kmeans.cluster_centers_, reglogit)
    if ret:
        print "Reform result to remove g {} from {}".format(idxbad,identity)
        x   = []
        lisa = []
        for idx,lab in enumerate(kmeans.labels_):
            if idxbad == lab:
                #print "Removed {}".format(lis[idx]["filename"])
                continue
            x.append(lis[idx]['rep'])
            lisa.append(lis[idx])
        X = np.array(x)
        kmeans = KMeans(n_clusters=nc, random_state=0).fit(X)
        return True,kmeans,lisa
    return False,kmeans,lis


def analisaNeibors(identity,reglogit=None,nc=10):
    sret = mdb.db.fs.files.find({'identity':identity}).sort("seq",1)
    info = queryId(identity)
    x   = []
    lis = []
    for dimg in sret:
        if u"angh{:03d}".format(dimg['seq']) in info.keys():
            x.append(info[u"rep{:03d}".format(dimg['seq'])])
            dimg['rep']=info[u"rep{:03d}".format(dimg['seq'])]
            lis.append(dimg)
    X = np.array(x)
    kmeans = KMeans(n_clusters=nc, random_state=0).fit(X)
    if reglogit is not None:
        ret,kmeans,lis=reformKmeans(kmeans,reglogit,lis,identity)
        #if ret:
        #    ret,kmeans,lis=reformKmeans(kmeans,reglogit,lis,identity)
    return kmeans.cluster_centers_,lis,kmeans.labels_
    #===========================================================================
    # print kmeans.labels_
    # gr0=kmeans.cluster_centers_[0]
    # gr1=kmeans.cluster_centers_[1]
    # for idx,rep in enumerate(X):
    #     dist0=calcDisR(rep,gr0)
    #     dist1=calcDisR(rep,gr1)
    #     print "idx: {:3d} dist0:{:4.3f} dist1:{:4.3f} g:{:2d}".format(idx,dist0,dist1,kmeans.labels_[idx])
    #===========================================================================
    #nbrs = NearestNeighbors(n_neighbors=5, algorithm='brute').fit(X)
    #distances, indices = nbrs.kneighbors(X)
    #print distances
    #print indices
    #return True
def geraSame(centros,X,Y):
    for idx1,c1 in enumerate(centros):
        for idx2,c2 in enumerate(centros):
            if idx1 <= idx2:
                continue
            dist = calcDisR(c1, c2)
            #print "same:{}".format(dist)
            X.append(np.array([dist]))
            Y.append(1)
    return X,Y

def geraOthers(centros1,centros2,X,Y):
    for c1 in centros1:
        for c2 in centros2:
            dist = calcDisR(c1, c2)
            #print "other:{}".format(dist)
            X.append(np.array([dist]))
            Y.append(0)
    return X,Y

def geraLogit(pessoas):
    chaves = pessoas.keys()
    X = []
    Y = []
    for idx1,identity in enumerate(chaves):
        centros,lista,rotulos=pessoas[identity]
        X,Y = geraSame(centros,X,Y)
        for idx2,identity2 in enumerate(chaves):
            if idx1<=idx2:
                continue
            centros2,lista,rotulos=pessoas[identity2]
            X,Y = geraOthers(centros,centros2,X,Y)
    print "{} valores e respostas {}".format(len(X),len(Y))
    logreg=LogisticRegression(C=1e5)
    logreg.fit(X,Y)
    a = logreg.coef_[0,0]
    b = logreg.intercept_[0]
    dis = -b/a
    print "a:{} b:{} disref:{}".format(a,b,dis)
    return logreg

def containIn(pessoas,centrosOther):
    chaves = pessoas.keys()
    isFound = False
    whoFound = -1
    for identity in chaves:
        centros,lista,rotulos=pessoas[identity]
        contap = 0
        contag  = 0
        contapa = 0
        for c1 in centros:
            for c2 in centrosOther:
                dist=calcDisR(c1, c2)
                pb=logreg.predict_proba(np.array([dist]).reshape(1,-1))
                contag += 1
                #print "p:{} d:{}".format(pb,dist)
                if pb[0,1] > 0.5:
                    contap += 1
        print "Contagem em {} com p:{} de g:{}".format(identity,contap,contag)
        if contap > (contag/4):
            if not isFound:
                isFound = True
                whoFound = identity
                contapa = contap
            else:
                print "Pertencente a dois grupos? {} {} acrescentando no que tiver mais comuns".format(whoFound,identity)
                if contapa > contap:
                    return True,whoFound
                else:
                    return True,identity
    return isFound,whoFound

#combina no objeto pessoa as informacoes
def mergeIn(pessoa,centros,lista,rotulos,reglogit,nc=10):
    x   = []
    lis = []
    for dimg in lista:
        x.append(dimg['rep'])
        lis.append(dimg)
    centros,lista,rotulos = pessoa
    for dimg in lista:
        x.append(dimg['rep'])
        lis.append(dimg)
    X = np.array(x)
    kmeans = KMeans(n_clusters=nc, random_state=0).fit(X)
    if reglogit is not None:
        ret,kmeans,lis=reformKmeans(kmeans,reglogit,lis,identity)
    return kmeans.cluster_centers_,lis,kmeans.labels_

#separa um elemento de cada grupo
def separateIm(lista,rotulos,nc=10):
    imgs={}
    for idx in range(nc):
        for idx2,rot in enumerate(rotulos):
            if idx == rot:
                img=mdb.readIm(lista[idx2])
                #print lista[idx]['filename']
                h,w,c = img.shape
                if h == 0 or w == 0:
                    print "Image {} storage with strange format {}x{}".format(lista[idx]["filename"],h,w)
                    continue
                if w > 120:
                    fr = 120.0/float(w)
                    #print "Redimensionando devido a {} com fator {}".format(w,fr)
                    try:
                        img = cv2.resize(img,(0,0),fx=fr,fy=fr)
                    except(cv2.error,'OpenCV Error'):
                        print "Error in resize image {} with shape: {} ".format(lista[idx]["filename"],img.shape)
                imgs[idx]=img
                break
    return imgs

#visualiza resultados da separacao
def viewResult(pessoas,associate,timeup=30):
    largeimg = np.zeros((780,1320,3),np.uint8)
    xpos = 0
    ypos = 0
    for identity in pessoas.keys():
        centros,lista,rotulos = pessoas[identity]
        print "Id: {} has {} images".format(identity,len(lista))
        imgs = separateIm(lista,rotulos)
        for key in imgs.keys():
            img = imgs[key]
            if args.align:
                bb = align.getLargestFaceBoundingBox(img)
                if bb is not None:
                    img = img[bb.top():bb.bottom(),bb.left():bb.right()].copy()
            h,w,c = img.shape
            largeimg[ypos:ypos+h,xpos:xpos+w]=img
            xpos += 120
        cv2.putText(largeimg,str(identity),(xpos,ypos+20),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,0,255),2)
        xpos += 40
        dypos = 0
        if identity in associate.keys():
            xpos = 720
            for nid in associate[identity]:
                cv2.putText(largeimg,str(nid),(xpos,ypos+dypos+20),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,0,255),2)
                if xpos < 1240:
                    xpos += 60
                else:
                    xpos = 720
                    dypos += 20
        xpos = 0
        if ypos < 600:
            ypos += 120
        else:
            cv2.imshow("Result",largeimg)
            if cv2.waitKey(timeup)& 0xFF == ord('q'):
                return True
            largeimg = np.zeros((780,1320,3),np.uint8)
            ypos = 0


    cv2.imshow("Result",largeimg)
    if cv2.waitKey(timeup)& 0xFF == ord('q'):
        return True
    return False

def saveToDb(pessoas):
    for key in pessoas.keys():
        centros,lista,rotulos=pessoas[key]
        for item in lista:
            mdb.db.resultado2.insert({'grobalid':key,'relativeid':item['identity'],'fid':item['_id']})

def cleanUpData(pessoa,reglogit,identity,nc=10):
    cluster_centers,lista,labels = pessoa
    ret,idxbad = anlyseCluster(cluster_centers, reglogit,trh=0.5)
    while ret:
        print "Reform result to remove g {} from {}".format(idxbad,identity)
        x   = []
        lisa = []
        for idx,lab in enumerate(labels):
            if idxbad == lab:
                #print "Removed {}".format(lis[idx]["filename"])
                continue
            x.append(lista[idx]['rep'])
            lisa.append(lista[idx])
        X = np.array(x)
        kmeans = KMeans(n_clusters=nc, random_state=0).fit(X)
        cluster_centers=kmeans.cluster_centers_
        lista=lisa
        labels=kmeans.labels_
        ret,idxbad = anlyseCluster(kmeans.cluster_centers_, reglogit,trh=0.5)
    return cluster_centers,lista,labels

if __name__ == '__main__':
    initMongoConn()

    ret = mdb.db.fs.files.aggregate([
                        #{"$match":{"base":args.base}},
                        {"$group":{"_id":{"identity":"$identity"},"count":{"$sum":1}}},
                        {"$sort": SON([("count", -1)])} #SON([("_id", 1)])}
                        ])
    conta = 0
    pessoas={}
    associate={}
    listamb = []
    for item in ret: #['result']:
        identity = item['_id']['identity']
        listamb.append(item)
        print "Id: {} count: {}".format(identity,item['count'])

    for item in listamb:
        identity = item['_id']['identity']
        if item['count'] < 80:
            continue
        print "Id: {} count: {}".format(identity,item['count'])
        if conta < 5:
            pessoas[identity]= analisaNeibors(item['_id']['identity'])
        else:
            #print "Agrupa"
            if conta <18:
                logreg=geraLogit(pessoas)
            if conta == 5:
                print "Cleanup data"
                for pk in pessoas.keys():
                    pessoas[pk]=cleanUpData(pessoas[pk],logreg,pk)
                logreg=geraLogit(pessoas)
            centros,lista,rotulos = analisaNeibors(item['_id']['identity'],logreg)
            isFound,whoFound=containIn(pessoas,centros)
            if isFound:
                pessoas[whoFound]=mergeIn(pessoas[whoFound],centros,lista,rotulos,logreg)
                if whoFound not in associate.keys():
                    associate[whoFound] = []
                associate[whoFound].append(identity)
            else:
                print "Nova identidade {}".format(identity)
                pessoas[identity]= centros,lista,rotulos
            if viewResult(pessoas,associate):
                break
        conta += 1
        #if conta > 400:
        #    cv2.waitKey(0)
        #    break
    viewResult(pessoas,associate,timeup=0)
    for pk in pessoas.keys():
        pessoas[pk]=cleanUpData(pessoas[pk],logreg,pk)
    viewResult(pessoas,associate,timeup=0)

    saveToDb(pessoas)
    if ws is not None:
        ws.close()

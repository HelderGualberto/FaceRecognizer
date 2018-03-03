#!/usr/bin/python

# Autor: Roberto Kenji Hiramatsu
# Copyright: Universidade de Sao Paulo/Huawei
# Agrupamento de grupos de refencia com NN4.smallv2
# Data: 2016/12/10

from libPoseDB import MongoConnOg
from sklearn.cluster import KMeans
import argparse
import os
import numpy as np
import cv2

parser = argparse.ArgumentParser()


parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join("models","dlib", "shape_predictor_68_face_landmarks.dat"))

parser.add_argument('--host', type=str, help="Host to mongodb",
                    #default="192.168.10.236")
                    default="kurentofront.pad.lsi.usp.br")

parser.add_argument('--port', type=int, help="Port of  mongodb",
                    default=37027)


args = parser.parse_args()

def showGrupo(grupo):
    largeimg = np.zeros((980,1200,3),np.uint8)
    xpos = 0
    ypos = 0
    for nome,tipo,arquivo in grupo:
        print "Carrega {}".format(arquivo)
        img = mdb.leArquivoST(arquivo)
        h,w,c = img.shape
        largeimg[ypos:ypos+h,xpos:xpos+w]=img
        cv2.putText(largeimg,nome,(xpos,ypos+20),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),2)
        cv2.putText(largeimg,tipo,(xpos,ypos+40),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),2)
        if xpos < 1080:
            xpos += 120
        else:
            xpos = 0
            if ypos < 700:
                ypos += 120
            else:
                cv2.imshow("Listagem",largeimg)
                if cv2.waitKey(0)& 0xFF == ord('q'):
                    return False
                largeimg = np.zeros((980,1200,3),np.uint8)
                xpos = 0
                ypos = 0

    cv2.imshow("Listagem",largeimg)
    if cv2.waitKey(0)& 0xFF == ord('q'):
        return False
    return True

if __name__ == '__main__':
    mdb = MongoConnOg(url="mongodb://{}:{}".format(args.host,args.port))
    pessoas = mdb.recupera()
    x     = []
    lista = []
    for nome in pessoas.keys():
        print nome
        for d in pessoas[nome]:
            x.append(d["rep"])
            lista.append((nome,'N',"ico.{}".format(d['arq'])))
            x.append(d["repFlip"])
            lista.append((nome,'F',"ico.{}".format(d['arq'])))

    X = np.array(x)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(X)

    grupos={}
    for idx,label in enumerate(kmeans.labels_):
        print "{} -> l: {}".format(lista[idx],label)
        if label not in grupos.keys():
            grupos[label]=[]
        grupos[label].append(lista[idx])

    for idx in grupos.keys():
        showGrupo(grupos[idx])




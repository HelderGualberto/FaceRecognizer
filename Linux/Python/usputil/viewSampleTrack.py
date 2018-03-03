#!/usr/bin/env python2
# Autor: Roberto Kenji Hiramatsu
#
# Copyright 2016 Universidade de Sao Paulo
#
# Visualizacao de dados da base coletada
#
# Data inicial: 2016-11-25 - inicial

import os
import libColetaFaceDB
from bson.son import SON
import cv2
import numpy as np
import argparse

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
                    default="10.20.0.253")

parser.add_argument('--port', type=int, help="Port of  mongodb",
                    #default=37027)
                    default=27017)

parser.add_argument('--base', type=str, help="Database on  mongodb",
                    #default="huaweicam")
                    #default="huaweicam2s")
                    #default="huaweicam3s")
                    #default="huaweicam4s")
                    default="huaweicam13s")


parser.add_argument('--align', type=bool, help="Alinhamento para entrada na rede neural",
                    default=False)

args = parser.parse_args()
mdb = libColetaFaceDB.MongoConn(url="mongodb://{}:{}".format(args.host,args.port),
                                dbs=args.base)


ret = mdb.db.fs.files.aggregate([
                        #{"$match":{"base":args.base}},
                        {"$group":{"_id":{"identity":"$identity"},"count":{"$sum":1}}}
                        ])
#,
#                        {"$sort": SON([("count", -1), ("_id", 1)])}
from align_dlib import AlignDlib
# alocado globalmente o extrador dlib
align = AlignDlib(args.dlibFacePredictor)


def apresenta(identity):
    sret = mdb.db.fs.files.find({'identity':identity})
    largeimg = np.zeros((960,1200,3),np.uint8)
    xpos = 0
    ypos = 0
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
        if xpos < 1080:
            xpos += 120
        else:
            xpos = 0
            if ypos < 720:
                ypos += 120
            else:
                cv2.putText(largeimg,str(identity),(0,ypos+20),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),3)
                cv2.imshow("Listagem",largeimg)
                if cv2.waitKey(0)& 0xFF == ord('q'):
                    return False
                ypos = 0
                largeimg = np.zeros((960,1200,3),np.uint8)
    cv2.putText(largeimg,str(identity),(0,ypos+20),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),3)
    cv2.imshow("Listagem",largeimg)
    if cv2.waitKey(0)& 0xFF == ord('q'):
        return False
    return True


for doc in ret:
    print "{} {}".format(doc["_id"]["identity"],doc["count"])
    if not apresenta(doc["_id"]["identity"]):
        break

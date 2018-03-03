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
                    default="192.168.10.236")

parser.add_argument('--port', type=int, help="Port of  mongodb",
                    default=37027)

parser.add_argument('--base', type=str, help="Database on  mongodb",
                    default="huaweicam")


args = parser.parse_args()
mdb = libColetaFaceDB.MongoConn(url="mongodb://{}:{}".format(args.host,args.port),
                                dbs=args.base)


#ret = mdb.db.fs.files.aggregate([
#                        {"$match":{"base":args.base}},
#                        {"$group":{"_id":{"identity":"$identity"},"count":{"$sum":1}}},
#                        {"$sort": SON([("count", -1), ("_id", 1)])}
#                        ])

ret = mdb.db.fs.files.find()

for doc in ret:
    #print "identity: {} file: {}".format(doc["identity"],doc["filename"])
    print doc
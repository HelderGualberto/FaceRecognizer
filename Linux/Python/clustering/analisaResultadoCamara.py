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

parser = argparse.ArgumentParser()


parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join("models","dlib", "shape_predictor_68_face_landmarks.dat"))

parser.add_argument('--host', type=str, help="Host to mongodb",
                    default="mongopad.pad.lsi.usp.br")

parser.add_argument('--port', type=int, help="Port of  mongodb",
                    default=37027)

parser.add_argument('--base', type=str, help="Database on  mongodb",
                    default="camara")


parser.add_argument('--align', type=bool, help="Alinhamento para entrada na rede neural",
                    default=False)


args = parser.parse_args()
mdb = libColetaFaceDB.MongoConn(url="mongodb://{}:{}".format(args.host,args.port),
                                dbs=args.base)


if __name__ == '__main__':
    ret = mdb.db.resultado.aggregate([
                        #{"$match":{"base":args.base}},
                        {"$group":{"_id":{"grobalid":"$grobalid"},"count":{"$sum":1}}},
                        {"$sort": SON([("count", -1)])} #SON([("_id", 1)])}
                        ])
    
    for item in ret: #['result']:
        identity = item['_id']['grobalid']
        print "id:{}  {:5d}".format(identity,item['count'])
        localret = mdb.db.resultado.find({'grobalid':identity}).sort("relativeid",1)
        for e in localret:
            retim = mdb.db.fs.files.find({'_id':e['fid']})
            for dimg in retim:
                img = mdb.readIm(dimg)
                cv2.imshow("{}.{}".format(identity,e['relativeid']),img)
            if cv2.waitKey(0)& 0xFF == ord('q'):
                break
        
           
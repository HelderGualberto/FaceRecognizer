#!/usr/bin/python

# Autor: Roberto Kenji Hiramatsu
# Copyright: Universidade de Sao Paulo/Huawei
# Biblioteca para interagir como mongo db de coleta de faces para processamento
# Data: 2016/10/10
#       2016/11/18 - Tratamento em memoria
#       2016/12/07 - Tratador de leitura de imagem modificado para tabalhar com remocao automatica e retorno diretamente

import pymongo
import gridfs
from pymongo import MongoClient
import cv2
#import cv2.cv as cv
import os,sys
import re

from pathlib import Path

from io import BytesIO

class MongoConn:
    def __init__(self,url="mongodb://10.0.0.236:37027",dbs="facecoleta"):
        self.client = MongoClient(url)
        self.db = self.client[dbs]
        self.fs = gridfs.GridFS(self.db)
        self.listaProcessada = None
        print "Conectado em {} na base {}".format(url,dbs)

    #implementado versao de ler arquivo de imagem compativel
    def readIm(self,d,path="img",remtemp=True):
        if not os.path.exists(path):
            print "Create {}".format(path)
            os.makedirs(path)
        else:
            if not os.path.isdir(path):
                print "{} is not a directory. Check please!".format(path)
                return None
        arquivo = open(os.path.join(path,d['filename']),"wb")
        arquivo.write(self.fs.get(d['_id']).read())
        arquivo.close()
        img = cv2.imread(os.path.join(path,d['filename']))
        if remtemp:
            os.remove(os.path.join(path,d['filename']))
        return img

    # Leitura controlada com remocao ao final da leitura
    # procura realizada pela chaves keys que sao passadas diretamente ao
    # localizacao dos arquivos
    def readImFromDB(self,keys,path="img",remtemp=True):
        ret = self.db.fs.files.find(keys)
        if not os.path.exists(path):
            print "Create {}".format(path)
            os.makedirs(path)
        else:
            if not os.path.isdir(path):
                print "{} is not a directory. Check please!".format(path)
                return None
        # lista para retornar a imagem e o documento relacionado na
        # base do mongodb
        imgs = []
        for d in ret:
            print "Lendo {}".format(d['filename'])
            arquivo = open(os.path.join(path,d['filename']),"wb")
            arquivo.write(self.fs.get(d['_id']).read())
            arquivo.close()
            img = cv2.imread(os.path.join(path,d['filename']))
            imgs.append((d,img))
            if remtemp:
                os.remove(os.path.join(path,d['filename']))
        return imgs


    #operacao para gravacao na base de referencia de face em coleta
    def gravaIm(self,img,base,seq,frame,datahora,depessoa,alinhado):
        buf = cv2.imencode(".jpg",img)
        #imgio = StringIO.StringIO()
        imgio = BytesIO()
        imgio.write(buf[1])
        imgio.seek(0)
        #now = datetime.now()
        yemoda = (datahora.year%100)*10000 + datahora.month*100 + datahora.day
        homi = datahora.hour*100 + datahora.minute
        arquivo = "{}_{:06d}_{:03d}_{:08d}_{:02d}{:02d}{:02d}{:02d}{:02d}.jpg".format(base,depessoa,seq,frame,
                                                                                  (datahora.year%100),datahora.month,
                                                                                  datahora.day,datahora.hour,
                                                                                  datahora.minute)
        try:
            fgrid = self.fs.new_file(filename=arquivo,identity=depessoa,base=base,
                                     seq=seq,frame=frame,yemoda=yemoda,homi=homi,alig=alinhado,tratado=False)
            fgrid.write(imgio)
        finally:
            fgrid.close()
        imgio.close()
        print "Enviado arquivo",arquivo


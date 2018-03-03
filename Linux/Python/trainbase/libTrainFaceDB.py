#!/usr/bin/python

# Autor: Roberto Kenji Hiramatsu
# Copyright: Universidade de Sao Paulo/Huawei
# Biblioteca para interagir como mongodb de imagens para processo de treinamento de rede neural
# Data: 2016/11/23 - Inicial para tratamento de leitura e escrita

import pymongo
import gridfs
from pymongo import MongoClient
import cv2
#import cv2.cv as cv
import os,sys
from bson.son import SON


from pathlib import Path

from io import BytesIO



class MongoConn:
    def __init__(self,url="mongodb://kurentofront.pad.lsi.usp.br:37027",dbs="trainbase"):
        self.client = MongoClient(url)
        self.db = self.client[dbs]
        self.fs = gridfs.GridFS(self.db)
        self.listaProcessada = None
        print "Conexao inicializada com {}".format(url)

    def readIm(self,path,d,remtemp=True):
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

    def readDBEntries(self,base):
        ret = self.db.fs.files.aggregate([
                        {"$match":{"base":base}},
                        {"$group":{"_id":{"identity":"$identity"},"count":{"$sum":1}}},
                        {"$sort": SON([("count", 1)])} #("count", -1),("_id", 1)
                        ])
        print "Terminado query de listagem de nomes"
        nameLists = {}
        for doc in ret:
            print "Processando para {}".format(doc["_id"]["identity"])
            items = self.db.fs.files.find({'identity':doc["_id"]["identity"],'base':base})
            listorgnames = []
            listadoc     = []
            for item in items:
                #print item["orgname"]
                listorgnames.append(item["orgname"])
                listadoc.append(item)            
            nameLists[doc["_id"]["identity"]]=(doc["count"],listorgnames,listadoc)
        return nameLists        

    #operacao para gravacao na base de referencia de face em base de treinamento
    def gravaIm(self,img,base,seq,depessoa,orgname,ext=".png"):
        if img is None:
            print "Arquivo {}{:4d} nao processado".format(depessoa,seq)
            return
        w,h,c = img.shape
        if w == 0 or h == 0:
            print "Arquivo {}{:4d} nao processado por conteudo vazio".format(depessoa,seq)
            return
        
        buf = cv2.imencode(ext,img)
        
        #imgio = StringIO.StringIO()
        imgio = BytesIO()
        imgio.write(buf[1])
        imgio.seek(0)
        #now = datetime.now()
        arquivo = "{}_{:04d}{}".format(depessoa,seq,ext)
        try:
            # hasRep - indicacao que esta tem representacao associada e tera os campos:
            #   - alimg - imagem alinhada para input formado por al<tamanho>_<arquivo>.png
            #   - rep   - vetor contendo a representacao da imagem para rede
            #   - tydn  - denominacao da rede neural usada para gerar a representacao
            fgrid = self.fs.new_file(filename=arquivo,identity=depessoa,base=base,
                                     seq=seq,width=w,height=h,orgname=orgname,hasRep=False)
            fgrid.write(imgio)
        finally:
            fgrid.close()
        imgio.close()
        print "Enviado arquivo",arquivo
    
    def __del__(self):        
        self.client.close()

                
            


#!/usr/bin/python

# Autor: Roberto Kenji Hiramatsu
# Copyright: Universidade de Sao Paulo/Huawei
# Biblioteca para interagir como mongo db
# Data: 2016/11/01 - baseado na libPoseDB para arqmazenar da camara dos deputados para

# Python 2/3 compatibility
from __future__ import print_function

import pymongo
import gridfs
from pymongo import MongoClient
import cv2
#import cv2.cv as cv
import os,sys
import re
import pickle
import os

from pathlib import Path

# classe abstrata para operacao dos dados
class MongoConn:
    def recupera(self):
        retcur = self.dbinfo.base.find()
        conta = 0
        pessoas = []
        conjunto = {}
        for document in retcur:
            if document["pessoa"] in pessoas:
                conjunto[document["pessoa"]].append(document)
            else:
                pessoas.append(document["pessoa"])
                conjunto[document["pessoa"]] = []
                conjunto[document["pessoa"]].append(document)
            #print document["arq"],document["pessoa"]
            conta += 1
        # mantem informacao do conjuto na classe para utilizacao posterior
        self.conjunto = conjunto
        return conjunto
    #recupera imagens de um individuo
    def recuperaImg(self,nome='kenji'):
        expressao = '\wico\..+'
        parte = re.compile(expressao)
        retcur = self.db.fs.files.find({'base':'baseRefFace','nomepessoa':nome},{'filename':1})
        listaArquivos = []
        for d in retcur:
            if parte.search(d['filename']):
                #print("{} e icone".format(d))
                entrada={}
                lista=d['filename'].split('.',2)
                #print (lista)
                tipo = lista[0].split('ico',1)
                entrada["tipo"]=tipo[0]
                entrada["indice"]=int(lista[1])
                entrada["refere"]=lista[2]
                entrada["filename"]=d['filename']
                #print (lista[0].split('ico',1))
                p=Path(os.path.join("img",d['filename']))
                if not p.is_file():
                    self.leArquivo(d)
                listaArquivos.append(entrada)
        #print (listaArquivos)
        return listaArquivos
            #else:
                #print(d)


    def leArquivo(self,d):
        arquivo = open(os.path.join("img",d['filename']),"wb")
        arquivo.write(self.fs.get(d['_id']).read())

    #arquivo especifico da base
    def leArquivoST(self,filename):
        caminho = os.path.join("img",filename)
        p=Path(caminho)
        if p.is_file():
             img = cv2.imread(caminho)
             return img
        retcur = self.db.fs.files.find({'base':'baseRefFace','filename':filename})
        for d in retcur:
            p=Path(os.path.join("img",d['filename']))
            if not p.is_file():
                self.leArquivo(d)
        if p.is_file():
             img = cv2.imread(caminho)
             return img
        return None

    def carregaListaProcessada(self):
        retcur  = self.dbinfo.base.find({},{"arq":1})
        self.listaProcessada=[]
        for d in retcur:
            self.listaProcessada.append(d["arq"])

    #
    def salvaLogit(self,logreg,nomearq='logreg.pkl'):
        with open(nomearq,'wb') as output:
            pickle.dump(logreg,output,pickle.HIGHEST_PROTOCOL)
        fileId = self.fs.put(open(nomearq,'rb'),filename=nomearq,base='baseRefFace')
            #os.remove(nomearq)

    # carrega o arquivo de classificador logit relacionado
    #
    def carregalogit(self,nomearq='logreg.pkl',anglim=30,isp5=False):
        if isp5:
            arquivo = 'p5a'+str(anglim)+nomearq
        else:
            arquivo = 'a'+str(anglim)+nomearq
        retcur = self.db.fs.files.find({'base':'baseRefFace','filename':arquivo})
        for d in retcur:
            print("{} {}".format(d['filename'],d["length"]))
            with open(arquivo,'wb') as output:
                output.write(self.fs.get(d['_id']).read())
            with open (arquivo,'rb') as entrada:
                logreg=pickle.load(entrada)
            #os.remove(nomearq)
            return logreg
        return None

# classe para colecao de arquivos sobre os dados da camara dos deputados
class MongoConnCamDep(MongoConn):
    def __init__(self,url="mongodb://kurentofront.pad.lsi.usp.br:37027"): #"mongodb://mdb:27017"
        self.client = MongoClient(url)
        self.db = self.client.camarafs
        self.dbinfo = self.client.camaradb
        self.fs = gridfs.GridFS(self.db)
        self.listaProcessada = None




def gravaIm(mdb,img,arquivo,depessoa,estag=False):
    if not estag:
        cv2.imwrite(arquivo,img)
    fileId = mdb.fs.put(open(arquivo,'rb'),filename=arquivo,nomepessoa=depessoa,base='baseRefFace')
    os.remove(arquivo)
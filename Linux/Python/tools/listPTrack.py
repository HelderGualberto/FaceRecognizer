#!/usr/bin/env python2
# Autor: Roberto Kenji Hiramatsu
#
# Copyright 2016 Universidade de Sao Paulo
#
# gerar lista para gerar pontos de track ou rastreamento das faces 
# Data inicial: 2016-01-20
from pymongo import MongoClient
client = MongoClient("mongodb://192.168.10.236:37027")
db = client["corredor"]

cols = db.collection_names()

def verifica(colec):
   retorna = True
   ret = db["p_{}".format(colec)].find()
   for ite in ret:
       if 'trackSeq' in ite.keys():
          retorna = False
   return retorna

#lista de colecao de frames e area da imagem com movimento
listaF = []
#lista de colecao de informacoes de possiveis faces (preprocessamento do haar)
listaC = []
#lista de faces encotradas com dlib
lista = []
for nc in cols:
   parts =nc.split('_')
   #print parts[1]
   # verifica se tem a parte frame ou cand
   if len(parts)>2:
      if parts[2] == 'frame':
         listaF.append(parts[1])
         #print "{} e tipo sequencia de frame".format(nc)
      elif parts[2] == 'cand':
         listaC.append(parts[1])
   else:
      lista.append(parts[1])
      #print "{} e tipo face".format(nc)
listaP = []
texto  = " "
for base in lista:
   listaP.append(base)
   if verifica(base):
      texto = "{} {} ".format(texto,base)
print texto

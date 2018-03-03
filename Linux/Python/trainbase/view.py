#!/usr/bin/python

# Autor: Roberto Kenji Hiramatsu
# Copyright: Universidade de Sao Paulo/Huawei
# Para carregar imagens para uso em treinamento de rede neural
# Data: 2016/11/23 - Inicial
#       2016/11/24 - com visualizador das imagens armazenadas em quadros de 120x120
import argparse
import os
from imutils import paths
import cv2
import numpy as np

import libTrainFaceDB

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str,
                    help="Diretorio contendo raiz das imagens a serem recuperadas da base",
                    #default="/srv/imagens_filtradas")
                    default="temp")

parser.add_argument('--base', type=str,
                    help="Nome da base / versao",
                    #default="/srv/imagens_filtradas")
                    default="f160621")

parser.add_argument('--host', type=str,
                    help="Nome da maquina que contem a base do mongodb",
                    #default="/srv/imagens_filtradas")
                    default="kurentofront.pad.lsi.usp.br")

parser.add_argument('--port', type=int,
                    help="Numero da porta do mongodb",
                    #default="/srv/imagens_filtradas")
                    default=37027)

parser.add_argument('--cleanup', type=bool,
                    help="Limpa diretorio temporario",
                    #default="/srv/imagens_filtradas")
                    default=True)

parser.add_argument('--summary', type=bool,
                    help="Limpa diretorio temporario",
                    #default="/srv/imagens_filtradas")
                    default=False)

args = parser.parse_args()

mdb = libTrainFaceDB.MongoConn(url="mongodb://{}:{}".format(args.host,args.port))

def drawPersonIm(nome,pathim,listadoc,cleanup):
    largeimg = np.zeros((960,1440,3),np.uint8)
    xpos = 0
    ypos = 0 
    for doc in listadoc:
        img=mdb.readIm(pathim,doc,remtemp=cleanup)
        h,w,c = img.shape
        if w > 120:
            fr = 120.0/float(w)
            img = cv2.resize(img,(0,0),fx=fr,fy=fr)
            h,w,c = img.shape
        largeimg[ypos:ypos+h,xpos:xpos+w]=img
        cv2.putText(largeimg,str(doc["seq"]),(xpos,ypos+40),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),1)        
        if xpos < 1320:            
            xpos += 120
        else:
            xpos = 0
            if ypos < 720:
                ypos += 120
            else:
                cv2.putText(largeimg,nome,(0,ypos+20),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),3)                
                cv2.imshow("Listagem",largeimg)
                if cv2.waitKey(0)& 0xFF == ord('q'):
                    return False                
                ypos = 0
                largeimg = np.zeros((960,1440,3),np.uint8)
    cv2.putText(largeimg,nome,(0,ypos+20),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),3)                
    cv2.imshow("Listagem",largeimg)
    if cv2.waitKey(0)& 0xFF == ord('q'):
        return False
    return True
        
if __name__=="__main__":    
    lista = mdb.readDBEntries(args.base)
    #criar diretorio temporario
    if not os.path.exists(args.dir):
            print "Create {}".format(args.dir)
            os.makedirs(args.dir)    
    conta   = 0
    contaim = 0   
    for nome in lista.keys():
        n,listaoldn,listadoc=lista[nome]
        conta   += 1
        contaim += n
        if args.summary:
            continue
        print "{} : {}".format(nome,n)
        pathim = os.path.join(args.dir,nome)
        if not drawPersonIm(nome, pathim, listadoc,args.cleanup):
            break
        
    print "Estao armazenadas {} individuos e do total de {} imagens".format(conta,contaim)
    if args.cleanup and not args.summary:
        print "Removendo diretorios temporarios em {}".format(args.dir)
        if os.path.isdir(args.dir):
            for subdir in os.listdir(args.dir):
                print "Removendo {}".format(os.path.join(args.dir,subdir))
                os.removedirs(os.path.join(args.dir,subdir))
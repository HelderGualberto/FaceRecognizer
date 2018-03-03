#!/usr/bin/python

# Autor: Roberto Kenji Hiramatsu
# Copyright: Universidade de Sao Paulo/Huawei
# Para carregar imagens para uso em treinamento de rede neural
# Data: 2016/11/23 - Inicial
import argparse
import os
from imutils import paths
import cv2

import libTrainFaceDB


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str,
                    help="Diretorio contendo raiz das imagens a serem armazenadas na base",
                    #default="/srv/imagens_filtradas")
                    default="D:\\projetos\\Safety_City_offdata\\bases\\usp\\filtrada\\imagens_filtradas")

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


args = parser.parse_args()

mdb = libTrainFaceDB.MongoConn(url="mongodb://{}:{}".format(args.host,args.port))

#metodo para gravar os aquivos de um individuo na base
def updata(identity,path,base,oldFname,separator="\\"):
    listIm = paths.list_images(path)
    contai = 0
    for fileim in listIm:
        vf = fileim.split(separator)
        #print vf[len(vf)-1]
        if oldFname is not None:
            if fileim in oldFname:
                print "O arquivo {} tem duplicacao".format(fileim)
                continue
        #print "Processa {}".format(fileim)
        img = cv2.imread(fileim)
        mdb.gravaIm(img, base, contai, identity,vf[len(vf)-1], ".png")
        contai += 1
    return contai

if __name__=="__main__":    
    print "Carregando arquivos de {}".format(args.dir)
    listagem=os.listdir(args.dir)
    conta  = 0
    contas = 0
    lista = mdb.readDBEntries(args.base)
    nomereg = lista.keys()
    for identidade in listagem:
        oldFname = None
        if nomereg is not None:
            if identidade in nomereg:
                n,oldFname,listadoc = lista[identidade]
        if os.path.isdir(os.path.join(args.dir,identidade)):
            contas += updata(identidade,os.path.join(args.dir,identidade),args.base,oldFname,separator=os.path.sep)
            conta += 1
        else:
            print "O caminho {} nao e diretorio".format(os.path.join(args.dir,identidade))
    
    print "Processado {} diretorios e {} arquivos".format(conta,contas)

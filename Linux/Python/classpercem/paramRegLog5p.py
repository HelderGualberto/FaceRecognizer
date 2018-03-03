#!/usr/bin/python

# Autor: Roberto Kenji Hiramatsu
# Copyright: Universidade de Sao Paulo/Huawei
# Script para gerar regressao logistica considerando so seguintes parametros:
# Angulo de enclinacao horizontal e vertical do rosto menor que 30 graus em relacao a referencia 0 0
# tomada dos valores em relacao a imagem base e a imagem espelhada - analise com todas as faces e valores de inclinacao
# Data: 2016/10/18

# Python 2/3 compatibility

from libPoseDB import MongoConnOg
from libPoseDB import MongoConnT1
import cv2
import numpy as np

import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#carregando da nuvem usp
#mdb = MongoConnOg(url="mongodb://kurentofront.pad.lsi.usp.br:37027")
#com versao treinada
mdb = MongoConnT1(url="mongodb://kurentofront.pad.lsi.usp.br:37027")

# limite de inclinacao horizontal
limiteaceita=30

# tratamento para a mesma imagem e equivalente espelhada
def geraLinhaEspelho(ds):
    respgl = []
    #dist = calcDistancia(ds["rep"],ds["repFlip"])
    rep  = ds["rep"]
    repF = ds["repFlip"]
    n    = len(rep)
    for idx in range(n):
        respgl.append(abs(rep[idx]-repF[idx]))
    dangh = 2*abs(ds["angH"])
    respgl.append(dangh)
    dangv = 0.0
    respgl.append(dangv)
    return respgl


# tratamento para a mesma imagem e equivalente espelhada
def geraLinhaDif(ds,ds2):
    respN   = []
    respglF1 = []
    respglF2 = []
    rep  = ds["rep"]
    repF = ds2["rep"]
    n    = len(rep)
    for idx in range(n):
        respN.append(abs(rep[idx]-repF[idx]))
    #dist = calcDistancia(ds["rep"],ds2["rep"])
    #respN.append(dist)

    rep  = ds["rep"]
    repF = ds2["repFlip"]
    n    = len(rep)
    for idx in range(n):
        respglF1.append(abs(rep[idx]-repF[idx]))
    #distR1 = calcDistancia(ds["rep"],ds2["repFlip"])
    #respglF1.append(distR1)
    rep  = ds["repFlip"]
    repF = ds2["rep"]
    n    = len(rep)
    for idx in range(n):
        respglF2.append(abs(rep[idx]-repF[idx]))
    #distR2 = calcDistancia(ds["repFlip"],ds2["rep"])
    #respglF2.append(distR2)
    dangh = abs(ds["angH"]-ds2["angH"])
    respN.append(dangh)
    danghF = abs(ds["angH"]+ds2["angH"])
    respglF1.append(danghF)
    respglF2.append(danghF)
    dangv = abs(ds["angV"]-ds2["angV"])
    respN.append(dangv)
    #print "{} {} {} {} {}".format(ds["arq"],ds2["arq"],ds["angV"],ds2["angV"],dangv)
    respglF1.append(dangv)
    respglF2.append(dangv)
    return respN,respglF1,respglF2

limiteaceita=45
# cria as parciais da tabela para a mesma pessoa
def parcialTab(pessoa):
    npe = len(pessoa)
    resp = []
    for idx in range(npe):
        if abs(pessoa[idx]["angH"]) > limiteaceita or abs(pessoa[idx]["angV"])>limiteaceita:
            continue
        respp1=geraLinhaEspelho(pessoa[idx])
        resp.append(respp1)
        for idx2 in range(idx+1,npe):
            if abs(pessoa[idx2]["angH"]) > limiteaceita or abs(pessoa[idx2]["angV"])>limiteaceita:
                continue
            respp2,respp3,respp4 = geraLinhaDif(pessoa[idx],pessoa[idx2])
            resp.append(respp2)
            resp.append(respp3)
            resp.append(respp4)
    return resp


# cria as parciais da tabela para a mesma pessoa
def parcialTab2D(pessoa1,pessoa2):
    resp = []
    for ds in pessoa1:
        if abs(ds["angH"]) > limiteaceita or abs(ds["angV"])>limiteaceita:
            continue
        for ds2 in pessoa2:
            if abs(ds2["angH"]) > limiteaceita or abs(ds2["angV"])>limiteaceita:
                continue
            respp2,respp3,respp4 = geraLinhaDif(ds,ds2)
            resp.append(respp2)
            resp.append(respp3)
            resp.append(respp4)
    return resp

# criar a tabela de valores X,Y
def criarTabela(pessoas):
    identidade = []
    for p in pessoas.keys():
        identidade.append(p)

    X=[]
    Y=[]
    n=len(identidade)
    for idx in range(n):
        # para tratar com a propria pessoa
        resp = parcialTab(pessoas[identidade[idx]])
        for r in resp:
            X.append(r)
            Y.append(1)
        #para tratar com os demais
        for idx2 in range(idx+1,n):
            resp=parcialTab2D(pessoas[identidade[idx]],pessoas[identidade[idx2]])
            for r in resp:
                X.append(r)
                Y.append(0)
    return X,Y


if __name__=="__main__":
    X,Y = criarTabela(pessoas)
    n = len(X)
    conta = 0
    #criar grupo de teste e grupo de treinamento com indice par para treinamento e indice impar para teste
    Xtre = []
    Ytre = []
    Xtes = []
    Ytes = []
    for idx in range(n):
        if (idx%2) == 0 :
            Xtre.append(np.array(X[idx]))
            Ytre.append(Y[idx])
        else:
            Xtes.append(np.array(X[idx]))
            Ytes.append(Y[idx])
        conta += 1
    print "{} linhas".format(conta)


    C = 1.0

    # Create different classifiers. The logistic regression cannot do
    # multiclass out of the box.
    #'p5a'+str(limiteaceita)+
    classifiers = {'l1logreg.pkl': LogisticRegression(C=C, penalty='l1'), #'L1 logistic'
               'l2logregovr.pkl': LogisticRegression(C=C, penalty='l2'), #L2 logistic (OvR)
               #'Linear SVC': SVC(kernel='linear', C=C, probability=True,
               #                  random_state=0),
               #    'L2 logistic (Multinomial)': LogisticRegression(
               # C=C, solver='lbfgs', multi_class='multinomial'), #logregmult.pkl
                  'logreg.pkl':LogisticRegression(C=1e5) #Logistic base logreg.pkl
               }

    n_classifiers = len(classifiers)

    for index, (name, logreg) in enumerate(classifiers.items()):
        #logreg = linear_model.LogisticRegression(C=1e5)
        # we create an instance of Neighbours Classifier and fit the data.
        logreg.fit(Xtre, Ytre)
        mdb.salvaLogit(logreg, nomearq=name,anglim=limiteaceita,isp5=True)

        nt = len(Xtes)
        conta = 0
        falha = 0
        print "{} {}".format(name,logreg.coef_)
        for idxt in range(nt):
            z=logreg.predict(Xtes[idxt].reshape(1,-1))
            probas = logreg.predict_proba(Xtes[idxt].reshape(1,-1))
            conta += 1
            if z[0] != Ytes[idxt]:
                falha += 1
                #print "I:{} O:{} E:{}  P:{}".format(Xtes[idxt],z[0],Ytes[idxt],probas)

        print "Falha {} do total de {} com taxa de {} em {}".format(falha,conta,(float(falha)/float(conta)),name)

#!/usr/bin/python

# Autor: Roberto Kenji Hiramatsu
# Copyright: Universidade de Sao Paulo/Huawei
# Script para gerar regressao para analise de angulo de inclinacao de rosto sobre a distancia euclidiana

# Data: 2016/10/20

# Python 2/3 compatibility

from libPoseDB import MongoConnOg
from libPoseDB import MongoConnT1
#import cv2
#import numpy as np

#import matplotlib.pyplot as plt
#from sklearn import linear_model
#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC

#carregando da nuvem usp
mdbOg = MongoConnOg(url="mongodb://kurentofront.pad.lsi.usp.br:37027")
#com versao treinada
mdbT1 = MongoConnT1(url="mongodb://kurentofront.pad.lsi.usp.br:37027")

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


#calcular a distancia
def calcDistancia(r1,r2):
    soma = 0.0
    for idx, val in enumerate(r1):
        pa=r1[idx]-r2[idx]
        soma += pa * pa
    return soma




# tratamento para a mesma imagem e equivalente espelhada
def geraLinhaEspelho(ds):
    respgl = []
    dist = calcDistancia(ds["rep"],ds["repFlip"])
    respgl.append(dist)
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
    dist = calcDistancia(ds["rep"],ds2["rep"])
    respN.append(dist)
    distR1 = calcDistancia(ds["rep"],ds2["repFlip"])
    respglF1.append(distR1)
    distR2 = calcDistancia(ds["repFlip"],ds2["rep"])
    respglF2.append(distR2)
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


# cria as parciais da tabela para a mesma pessoa
def parcialTab(pessoa):
    npe = len(pessoa)
    resp = []
    #inicializar com maior angulo para determnar o menor angulo com vertical abaixo de 15 graus
    minangh = 45
    idxminangh = 0
    for idx in range(npe):
        if minangh > abs(pessoa[idx]["angH"]) and pessoa[idx]["angV"] > 0:
            idxminangh = idx
            minangh = abs(pessoa[idx]["angH"])
            print "min em H:{} V:{}".format(pessoa[idx]["angH"],pessoa[idx]["angV"])

    ds = pessoa[idxminangh]
    for idx in range(npe):
        #desconsiderar a distancia do menor
        if idxminangh != idx:
            ds2=pessoa[idx]
            dist=calcDistancia(ds["rep"],ds2["rep"])
            difh = abs(ds["angH"]-ds2["angH"])
            difv = ds["angV"]-ds2["angV"]
            resp.append((difh,difv,dist))
            print "{:5.1f} ;{:5.1f} ; {:6.2f}".format(difh,difv,dist)
            distF=calcDistancia(ds["repFlip"],ds2["repFlip"])
            resp.append((difh,difv,distF))
            print "{:5.1f} ;{:5.1f} ; {:6.2f}".format(difh,difv,distF)
        #=======================================================================
        # respp1=geraLinhaEspelho(pessoa[idx])
        # resp.append(respp1)
        # for idx2 in range(idx+1,npe):
        #     respp2,respp3,respp4 = geraLinhaDif(pessoa[idx],pessoa[idx2])
        #     resp.append(respp2)
        #     resp.append(respp3)
        #     resp.append(respp4)
        #=======================================================================
    return resp




# criar a tabela de valores X,Y
def criarTabela(pessoas):
    identidade = []
    for p in pessoas.keys():
        identidade.append(p)

    X_a=[]
    Y_a=[]
    Z_a=[]
    X_r = []
    y_r = []
    n=len(identidade)
    for idx in range(n):
        print "Processando para {}".format(identidade[idx])
        # para tratar com a propria pessoa
        resp = parcialTab(pessoas[identidade[idx]])
        for x,y,z in resp:
            if x > 50 or y > 50:
                continue
            X_a.append(x)
            Y_a.append(y)
            Z_a.append(z)
            X_r.append([x,y])
            y_r.append(z)


    return np.array(X_a),np.array(Y_a),np.array(Z_a),np.array(X_r),np.array(y_r)

if __name__=="__main__":
    grau = 2
    # carregar as informacoes da base
    pessoas = mdbOg.recupera()
    X,Y,Z,X_r,y_r = criarTabela(pessoas)
    modelOg = make_pipeline(PolynomialFeatures(grau), Ridge())
    modelOg.fit(X_r,y_r)

    pessoas = mdbT1.recupera()
    X_t1,Y_t1,Z_t1,X_rt,y_rt = criarTabela(pessoas)
    modelT1 = make_pipeline(PolynomialFeatures(grau), Ridge())
    modelT1.fit(X_rt,y_rt)


    #===========================================================================
    # mpl.rcParams['legend.fontsize'] = 10
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.scatter(X, Y, Z, color='black')
    #===========================================================================

#===============================================================================
#     ah_plot = np.linspace(0,50,11)
#     av_plot = np.zeros(11)
#     z_plot = []
#     for idx,ah in enumerate(ah_plot):
#         z = modelOg.predict(np.array([ah,av_plot[idx]]).reshape(1,-1))
#         print "{:4.1f} ; {:5.2f}".format(ah,z[0])
#         z_plot.append(z[0])
#     z_plot = np.array(z_plot)
#
#     z_plott = []
#     for idx,ah in enumerate(ah_plot):
#         z = modelT1.predict(np.array([ah,av_plot[idx]]).reshape(1,-1))
#         print "{:4.1f} ; {:5.2f}".format(ah,z[0])
#         z_plott.append(z[0])
#     z_plott = np.array(z_plott)
#
#     plt.scatter(X, Z,  color='black',marker='o',label="nn4.smallv2")
#     plt.plot(ah_plot, z_plot, color='blue',linewidth=3,label="nn4.smallv2")
#     #plt.legend(loc='ang h x dist nn4.smallv2')
#
#     plt.scatter(X_t1, Z_t1,  color='gray',marker='^',label="nn4.usp0617")
#     plt.plot(ah_plot, z_plott, color='green',linewidth=3,label="nn4.usp0617")
#
#
#     plt.grid(True)
#     plt.axhline(0, color='black', lw=2)
#
#     plt.legend()
#===============================================================================

    ah_plot = np.zeros(11)
    av_plot = np.linspace(0,50,11)
    z_plot = []
    for idx,ah in enumerate(ah_plot):
        z = modelOg.predict(np.array([ah,av_plot[idx]]).reshape(1,-1))
        print "{:4.1f} ; {:5.2f}".format(av_plot[idx],z[0])
        z_plot.append(z[0])
    z_plot = np.array(z_plot)

    z_plott = []
    for idx,ah in enumerate(ah_plot):
        z = modelT1.predict(np.array([ah,av_plot[idx]]).reshape(1,-1))
        print "{:4.1f} ; {:5.2f}".format(av_plot[idx],z[0])
        z_plott.append(z[0])
    z_plott = np.array(z_plott)

    plt.scatter(Y, Z,  color='black',marker='o',label="nn4.smallv2")
    plt.plot(av_plot, z_plot, color='blue',linewidth=3,label="nn4.smallv2")
    #plt.legend(loc='ang h x dist nn4.smallv2')

    plt.scatter(Y_t1, Z_t1,  color='gray',marker='^',label="nn4.usp0617")
    plt.plot(av_plot, z_plott, color='green',linewidth=3,label="nn4.usp0617")


    plt.grid(True)
    plt.axhline(0, color='black', lw=2)

    plt.legend()

    #plt.xticks(())
    #plt.yticks(())
    #print z_plot
    #ax.plot(ah_plot,av_plot,z_plot,label='ang v=0')

    #===========================================================================
    # ah_plot = np.linspace(0,45,10)
    # av_plot = np.linspace(0,45,10)
    # z_plot = []
    # for idx,ah in enumerate(ah_plot):
    #     z = model.predict(np.array([ah,av_plot[idx]]).reshape(1,-1))
    #     print "{:4.1f} ; {:4.1f} {:5.2f}".format(ah,av_plot[idx],z[0])
    #     z_plot.append(z[0])
    # z_plot = np.array(z_plot)
    # #print z_plot
    # ax.plot(ah_plot,av_plot,z_plot,label='ang h=v')
    #===========================================================================

    #===========================================================================
    # ah_plot = np.zeros(10)
    # av_plot = np.linspace(0,45,10)
    # z_plot = []
    # for idx,ah in enumerate(ah_plot):
    #     z = modelOg.predict(np.array([ah,av_plot[idx]]).reshape(1,-1))
    #     print "{:4.1f} ; {:5.2f}".format(av_plot[idx],z[0])
    #     z_plot.append(z[0])
    # z_plot = np.array(z_plot)
    #===========================================================================
    #print z_plot

#===============================================================================
#     ax.plot(ah_plot,av_plot,z_plot,label='ang h=0')
#
#     ax.legend()
#
#     ax.set_xlabel('ang H')
#     ax.set_ylabel('ang V')
#     ax.set_zlabel('distance')
#===============================================================================
    plt.show()

#!/usr/bin/python

# Autor: Roberto Kenji Hiramatsu
# Copyright: Universidade de Sao Paulo/Huawei
# Script para analisar imagens armazenadas na base mongo db
# Data: 2016/10/06

# Python 2/3 compatibility

from libPoseDB import MongoConnOg
import cv2
import numpy as np


#carregando da nuvem usp
mdb = MongoConnOg(url="mongodb://kurentofront.pad.lsi.usp.br:37027")

#calcular a distancia
def calcDistancia(r1,r2):
    soma = 0.0
    for idx, val in enumerate(r1):
        pa=r1[idx]-r2[idx]
        soma += pa * pa
    return soma

#indice de compressao/qualidade do jpeg
qualityf = [90 ,  80 ,  70 ,  50 , 33 ,25 , 10 , 5 ]
#indice de niveis de bit sendo respectivamente 7 6 5 4 e 3
vmask = [ 1 , 3 , 7 , 15 , 31]

# carregar as informacoes da base
pessoas = mdb.recupera()
# carregar a lista de imagens relacionadas
arquivos={}
for p in pessoas.keys():
    arquivos[p]=mdb.recuperaImg(p)


def desenhaem(largeimg,ypos,file1,file2,texto,flip=False):
    xpos=0
    img1 = cv2.imread("img\\"+file1)
    [h,w,c] = img1.shape
    largeimg[ypos:ypos+h,xpos:xpos+w]=img1
    xpos += w
    img2 = cv2.imread("img\\"+file2)
    if flip:
        img2 = cv2.flip(img2,1)
    largeimg[ypos:ypos+h,xpos:xpos+w]=img2
    xpos += w
    ypos += h
    cv2.putText(largeimg, texto, (xpos,ypos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0,0), 2)
    return True if ypos+h > 960 else False, ypos

def analisar45(pessoa):
    #separar as imagens laterais direita e esquerda
    listae = []
    listad = []
    listac = []
    largeimg = np.zeros((960,1440,3), np.uint8)
    largeimg[:,:] =(255,255,255)
    ypos=0

    for item in pessoa:
        #carrega icone do arquivo
        mdb.leArquivoST("ico."+item["arq"])
        try:
            if item["angH"] < -30:
                listae.append(item)
                #print (item["arq"],item["angH"])
            else:
                if item["angH"] < 30:
                    listac.append(item)

            if item["angH"] > 30:
                listad.append(item)
                #print (item["arq"],item["angH"])
        except TypeError:
            print (item)

    if len(listae)>0 and len(listad)>0:
        for esq in listae:
            for dir in listad:
                dist = calcDistancia(esq["rep"],dir["repFlip"])
                distR = calcDistancia(esq["repFlip"],dir["rep"])
                dfa=abs(dir["angH"]-abs(esq["angH"]))
                media = abs(dir["angH"]+abs(esq["angH"]))/2
                dfav=abs(abs(dir["angV"])-abs(esq["angV"]))
                mediav = abs(abs(dir["angV"])+abs(esq["angV"]))/2
                print "{} ; {} ; {} ; {} ; {} ; {} ; {} ; ed".format(esq["arq"],dir["arq"],dist,dfa,media,dfav,mediav)
                print "{} ; {} ; {} ; {} ; {} ; {} ; {} ; de".format(dir["arq"],esq["arq"],distR,dfa,media,dfav,mediav)
                texto="DE dist:{:5.3f} dif h:{:4.1f} med:{:4.1f} dif v:{:4.1f} distR:{:5.3f}".format(dist,dfa,media,dfav,distR)
                ret,ypos = desenhaem(largeimg,ypos,"ico."+esq["arq"],"ico."+dir["arq"],texto,flip=True)
                if ret:
                    cv2.imshow("Resultado",largeimg)
                    cv2.waitKey(0)
                    ypos=0
                    largeimg = np.zeros((960,1440,3), np.uint8)
                    largeimg[:,:] =(255,255,255)
    if len(listae)>1:
        n = len(listae)
        for idx in range(n):
            for idx2 in range((idx+1),n):
                #print "1:{} 2:{}".format(listae[idx]["angH"],listae[idx2]["angH"])
                dist = calcDistancia(listae[idx]["rep"],listae[idx2]["rep"])
                dfa=abs(abs(listae[idx]["angH"])-abs(listae[idx2]["angH"]))
                media = abs(abs(listae[idx]["angH"])+abs(listae[idx2]["angH"]))/2
                dfav=abs(abs(listae[idx]["angV"])-abs(listae[idx2]["angV"]))
                mediav = abs(abs(listae[idx]["angV"])+abs(listae[idx2]["angV"]))/2
                print "{} ; {} ; {} ; {} : {} ; {} : {} ; ee".format(listae[idx]["arq"],listae[idx2]["arq"],dist,dfa,media,dfav,mediav)
                texto="EE dist:{:5.3f} dif h:{:4.1f} med:{:4.1f} dif v:{:4.1f}".format(dist,dfa,media,dfav)
                ret,ypos = desenhaem(largeimg,ypos,"ico."+listae[idx]["arq"],"ico."+listae[idx2]["arq"],texto)
                if ret:
                    cv2.imshow("Resultado",largeimg)
                    cv2.waitKey(0)
                    ypos=0
                    largeimg = np.zeros((960,1440,3), np.uint8)
                    largeimg[:,:] =(255,255,255)
    if len(listad)>1:
        n = len(listad)
        for idx in range(n):
            for idx2 in range((idx+1),n):
                dist = calcDistancia(listad[idx]["rep"],listad[idx2]["rep"])
                dfa=abs(listad[idx]["angH"]-abs(listad[idx2]["angH"]))
                media=abs(listad[idx]["angH"]+abs(listad[idx2]["angH"]))/2
                dfav=abs(abs(listad[idx]["angV"])-abs(listad[idx2]["angV"]))
                mediav=abs(abs(listad[idx]["angV"])+abs(listad[idx2]["angV"]))/2
                print "{} ; {} ; {} ; {} : {} ; {} : {} ; dd".format(listad[idx]["arq"],listad[idx2]["arq"],dist,dfa,media,dfav,mediav)
                texto="DD dist:{:5.3f} dif h:{:4.1f} med:{:4.1f} dif v:{:4.1f}".format(dist,dfa,media,dfav)
                ret,ypos = desenhaem(largeimg,ypos,"ico."+listad[idx]["arq"],"ico."+listad[idx2]["arq"],texto)
                if ret:
                    cv2.imshow("Resultado",largeimg)
                    cv2.waitKey(0)
                    ypos=0
                    largeimg = np.zeros((960,1440,3), np.uint8)
                    largeimg[:,:] =(255,255,255)

    for it in listac:
        dist = calcDistancia(it["rep"],it["repFlip"])
        dfa =2*abs(it["angH"])
        print "{} ; {} ; {} ; {} ; 0.0 ; {} ; {} ; cf".format(it["arq"],it["arq"],dist,dfa,it["angV"],it["angV"])
        texto="CF dist:{:5.3f} dif ang:{:4.1f} medv:{:4.1f}".format(dist,dfa,abs(it["angV"]))
        ret,ypos = desenhaem(largeimg,ypos,"ico."+it["arq"],"ico."+it["arq"],texto,flip=True)
        if ret:
            cv2.imshow("Resultado",largeimg)
            cv2.waitKey(0)
            ypos=0
            largeimg = np.zeros((960,1440,3), np.uint8)
            largeimg[:,:] =(255,255,255)

    cv2.imshow("Resultado",largeimg)
    cv2.waitKey(0)

for   pk in pessoas.keys():
    #print pk
    analisar45(pessoas[pk])

#===============================================================================
# # para listar amostra
# if __name__=="__main__":
#     pessoas = mdb.recupera()
#     arquivos={}
#     for p in pessoas.keys():
#         arquivos[p]=mdb.recuperaImg(p)
#         xpos=0
#         ypos=0
#         conta = 0
#         # loop over the input images
#         largeimg = np.zeros((960,1440,3), np.uint8)
#         for a in arquivos[p]:
#             img = cv2.imread(a["filename"])
#             [h,w,c] = img.shape
#             largeimg[ypos:ypos+h,xpos:xpos+w]=img
#             xpos += 96
#             if xpos >= 1440:
#                 xpos = 0
#                 ypos += 96
#             conta += 1
#             if conta == 150:
#                 cv2.imshow("Image", largeimg)
#                 key = cv2.waitKey(0)
#                 xpos = 0
#                 ypos = 0
#                 conta = 0
#         cv2.imshow("Image", largeimg)
#         key = cv2.waitKey(0)
#===============================================================================


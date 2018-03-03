#!/usr/bin/python

# Autor: Roberto Kenji Hiramatsu
# Copyright: Universidade de Sao Paulo/Huawei
# Analise de qualidade devido ao tamanho da imagem
# Data: 2017/01/01 - Inicial


import cv2 
import argparse
import openface
from quality import extractFaceDLibAli
from miscsolve import calcDisR
from imutils import paths
from os.path import join
from pose import solvePnPHPAng,calcHVAngRosto

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", 
            default="../../../imagens/qualityimg",
    help="path to input directory of images")
args = ap.parse_args()

net = openface.TorchNeuralNet(join("..","..","data","models", 'openface', #'treinado-jun16.t7'))
                                         'nn4.small2.v1.t7')
                              ,imgDim=96,
                              cuda=False)
def drawLandMS(img,landms,txt):
    for p in landms:
        cv2.circle(img, p, 2, (0,255,0), -1)
    cv2.imshow(txt,img)
    
def avalia(img,wref,repref,landmarksref,pg=0.97):
    fator = pg
    w = wref
    href,wref,c = img.shape
    anghref,angvref = solvePnPHPAng(landmarksref,(wref,href))
    angcabref,angvcabref,pp=calcHVAngRosto(landmarksref)
    print "H:{} V:{} x H:{} V:{}".format(anghref,angvref,angcabref,angvcabref)
    cv2.imshow("Amostra",img)
    k=cv2.waitKey(0)
    while w > 48: #48
        imgred = cv2.resize(img,(0,0),fx=fator,fy=fator)
        ret,imgAli,w,landms = extractFaceDLibAli(imgred)        
        if ret:
            rep  = net.forward(imgAli)
            dist = calcDisR(rep,repref)
            he,wi,c = imgred.shape
            angh,angv = solvePnPHPAng(landms,(wi,he))
            angcab,angvcab,pp=calcHVAngRosto(landms)
            difah  = (anghref-angh)
            difav  = (angvref-angv)
            difahs = (angcabref-angcab)
            difavs = (angvcabref-angvcab)
            print "{:03d}; {:03d}; {:03d}; {:5.3f}; {:5.3f}; {:5.1f}; {:5.1f}; {:5.1f}; {:5.1f}".format(w,wi,he,fator,dist,
                                                                                          difah, #
                                                                                          difav, #
                                                                                          difahs, #
                                                                                          difavs) #
            mdifa = max((abs(difah),abs(difav),abs(difahs),abs(difavs)))
            if dist>0.25 or mdifa > 15:
                drawLandMS(img.copy(),landmarksref,'Or')
                drawLandMS(imgred,landms,'Rd')
                k=cv2.waitKey(0)
        fator *= pg

if __name__=="__main__":
    conta = 0
    for imagePath in paths.list_images(args.images):
        img = cv2.imread(imagePath)
        ret,imgAli,w,landms = extractFaceDLibAli(img)
        if ret :
            rep  = net.forward(imgAli)
            print "Ref:{} -> {}".format(imagePath,w)
            avalia(img,w,rep,landms)
        else:
            print imagePath
        conta += 1
        if conta > 10:
            break
#!/usr/bin/env python2
# Autor: Roberto Kenji Hiramatsu
#
# Copyright 2016 Universidade de Sao Paulo
#
# Gera marcacao de sequencia de faces do mesmo individuo
# Data inicial: 2017-01-
from pymongo import MongoClient
import pymongo
import argparse

from bson.son import SON
import cv2
from os.path import join
import numpy as np
from miscsolve import rectXYHW
from quality import extractFaceDLib,variance_of_laplacian,fftScore
from trackingSimple import TrackingFaces

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", 
            default=" /vmstor/openface/SafetyCity/videos/sequence/p.1222023104",
    help="path to input directory of images")
ap.add_argument("-u", "--mdburl",
default="mongodb://192.168.10.236:37027",
    help="URL de conecao ao banco de dados")
ap.add_argument("-b", "--base",
default="p_1222023104",
#default="p_1222084100",
    help="URL de conecao ao banco de dados")
args = ap.parse_args()

client = MongoClient(args.mdburl)
db = client["corredor"]
colecao = db[args.base]
colecaoCan = db["{}_cand".format(args.base)]

def extractFrameIdx(filen):
        parta = filen.split('.')
        partb = parta[0].split('-')
        idx = int(partb[1])
        return idx

def createIdx(r):
    a,b,c,d = r
    return "{}.{}.{}.{}".format(a,b,c,d)

def drawResult(listaRel,listaF):
    largeimg = np.zeros((960,1440,3), np.uint8)
    xpos = 0
    ypos = 40
    texto =  "{} -> {}".format(listaF.identity,len(listaF.listHistory))
    for idxr,p in listaF.listHistory:        
        img = cv2.imread(join(args.images,listaRel[idxr][u'fileOut'])) 
        if img is None:
            print "{} in {} not found".format(listaRel[idxr][u'fileOut'],listaRel[idxr][u'r0'])
            continue
        ret,imgAli,we =extractFaceDLib(img)
        if not ret:
            imgAli = img
        gray = cv2.cvtColor(imgAli, cv2.COLOR_BGR2GRAY)
        indexLap = variance_of_laplacian(gray)
        indexFFT = fftScore(gray)           
        h,w,c = imgAli.shape
        if xpos+w < 1440:
            if ypos + h < 960:
                largeimg[ypos:ypos+h,xpos:xpos+w]=imgAli
                cv2.putText(largeimg,"{}x{}".format(w,h) ,      (xpos+ 5, ypos+  60),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                if indexLap < 125:
                    cv2.putText(largeimg,"{:.0f}".format(indexLap) ,(xpos+ 5, ypos+  80),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                else:
                    cv2.putText(largeimg,"{:.0f}".format(indexLap) ,(xpos+ 5, ypos+  80),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                if indexFFT < 0.125:
                    cv2.putText(largeimg,"{:3.2f}".format(indexFFT),(xpos+ 5, ypos+ 100),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                else:
                    cv2.putText(largeimg,"{:3.2f}".format(indexFFT),(xpos+ 5, ypos+ 100),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                xpos += w
            else:
                cv2.putText(largeimg,texto , (5, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                cv2.imshow("Resultado",largeimg)
                k=cv2.waitKey(0)
                largeimg = np.zeros((960,1440,3), np.uint8)
                xpos = 0
                ypos = 40
                largeimg[ypos:ypos+h,xpos:xpos+w]=imgAli
                cv2.putText(largeimg,"{}x{}".format(w,h) ,      (xpos+ 5, ypos+  60),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                if indexLap < 125:
                    cv2.putText(largeimg,"{:.0f}".format(indexLap) ,(xpos+ 5, ypos+  80),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                else:
                    cv2.putText(largeimg,"{:.0f}".format(indexLap) ,(xpos+ 5, ypos+  80),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                if indexFFT < 0.125:
                    cv2.putText(largeimg,"{:3.2f}".format(indexFFT),(xpos+ 5, ypos+ 100),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                else:
                    cv2.putText(largeimg,"{:3.2f}".format(indexFFT),(xpos+ 5, ypos+ 100),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                xpos += w
        else:
            xpos = 0 
            ypos += 120
            if ypos + h < 960:
                largeimg[ypos:ypos+h,xpos:xpos+w]=imgAli
                cv2.putText(largeimg,"{}x{}".format(w,h) ,      (xpos+ 5, ypos+  60),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                if indexLap < 125:
                    cv2.putText(largeimg,"{:.0f}".format(indexLap) ,(xpos+ 5, ypos+  80),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                else:
                    cv2.putText(largeimg,"{:.0f}".format(indexLap) ,(xpos+ 5, ypos+  80),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                if indexFFT < 0.125:
                    cv2.putText(largeimg,"{:3.2f}".format(indexFFT),(xpos+ 5, ypos+ 100),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                else:
                    cv2.putText(largeimg,"{:3.2f}".format(indexFFT),(xpos+ 5, ypos+ 100),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                xpos += w
            else:
                cv2.putText(largeimg,texto , (5, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                cv2.imshow("Resultado",largeimg)
                k=cv2.waitKey(0)
                largeimg = np.zeros((960,1440,3), np.uint8)
                xpos = 0
                ypos = 40
                largeimg[ypos:ypos+h,xpos:xpos+w]=img
    cv2.putText(largeimg,texto , (5, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    cv2.imshow("Resultado",largeimg)
    k=cv2.waitKey(0)
    
def updateSeq(listaRel,listaF):
    for idxr,p in listaF.listHistory:
        colecaoCan.update({'_id':listaRel[idxr][u'_id']}, {"$set": {'trackSeq':listaF.identity}}, upsert=False)
   
if __name__=="__main__":
    print args
    partc = args.images.split('\\')
    princ=partc[len(partc)-1]
    print princ
    ret = colecaoCan.find().sort([(u'fileIn',pymongo.ASCENDING)])
    
    previdx = 0
    previtem = None
    trackingF = TrackingFaces()
    rects = []
    listaRel = {}
    conta = 0 
    for item in ret:
        idx = extractFrameIdx(item['fileIn'])
        if idx == previdx:
            if 'r0' in item.keys():
               rects.append((conta,item['r0']))
               listaRel[conta]=item
            else:
               print item
            if 'r1' in item.keys():
               conta += 1
               rects.append((conta,item['r1']))
               listaRel[conta]=item
               #print item
            if 'r2' in item.keys():
               conta += 1
               rects.append((conta,item['r2']))
               listaRel[conta]=item
               #print item

        else:
            compList = trackingF.updateFromRectWH(previdx,rects)
            if len(compList) > 0 :
                for listaF in compList:
                    #updateSeq(listaRel,listaF)
                    print listaF                    
                    #drawResult(listaRel,listaF)
                        #del listaRel[createIdx(p)]                        
                        #print p
            rects=[]
            #listaRel = {}
            if 'r0' in item.keys():
               rects.append((conta,item['r0']))
               listaRel[conta]=item
            else:
               print item
            if 'r1' in item.keys():
               conta += 1
               rects.append((conta,item['r1']))
               listaRel[conta]=item
               #print item
            if 'r2' in item.keys():
               conta += 1
               rects.append((conta,item['r2']))
               listaRel[conta]=item
               #print item

        previdx = idx
        previtem = item
        conta += 1 
    ordenadoConfli = sorted(trackingF.conflitIdList)
    for cf in ordenadoConfli:
        print cf

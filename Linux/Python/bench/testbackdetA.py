#!/usr/bin/env python2
# Autor: Roberto Kenji Hiramatsu
#
# Copyright 2016 Universidade de Sao Paulo
#
# Teste para verificar acesso a stream de video rstp
# Data inicial: 2016-08-17

import numpy as np
import cv2
from imutils import paths
import argparse
import os
from common import clock, draw_str

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str,
                    help="url do video",
                    default='../corredor/corredor-1223084200.mp4')
                    #default='rtsp://admin:B30cd4Ro@127.0.0.1:8554/LiveMedia/ch1/Media2')
                    #default='rtsp://admin:B30cd4Ro@127.0.0.1:8554/LiveMedia/ch1/Media1')

args = parser.parse_args()


# classe para aglutina os retangulos processados
class SuperRect:
    def __init__(self,rect,idx):
        self.grect = []
        self.grect.append(rect)
        x,y,w,h = rect
        self.x0 = x
        self.y0 = y
        self.w = w
        self.h = h
        self.x1 = x + w
        self.y1 = y + h
        self.idx = idx

    def contains(self,rect):
        ox0,oy0,w,h = rect
        ox1 = ox0 + w
        oy1 = oy0 + h
        if self.x0 > ox1 or ox0 > self.x1:
            return False
        if self.y0 > oy1 or oy0 > self.y1:
            return False
        self.x0 = self.x0 if self.x0 < ox0 else ox0
        self.y0 = self.y0 if self.y0 < oy0 else oy0
        self.x1 = self.x1 if self.x1 > ox1 else ox1
        self.y1 = self.y1 if self.y1 > oy1 else oy1
        self.w = self.x1 - self.x0
        self.h = self.y1 - self.y0
        self.grect.append(rect)
        return True

    def getRect(self):
        return self.x0,self.y0,self.w,self.h

    # ajustar os retangulos aos limites minimos e maximos e retorna os pontos (x0,y0) (x1,y1)
    def getAjRect(self,w,h):
        self.x0 = self.x0 if self.x0 > 0 else 0
        self.y0 = self.y0 if self.y0 > 0 else 0
        self.x1 = self.x1 if self.x1 < w else w
        self.y1 = self.y1 if self.y1 < h else h
        #self.w = self.x1 - self.x0
        #self.h = self.y1 - self.y0
        return (self.x0,self.y0,self.x1,self.y1)

#funcao para aglutinar os subretangulos contidos em listc e retorna a lista se SuperRect aglutinadas
def aglutina(listc):
    lists = []
    listarem = []
    conta = 0
    #ordena os retangulos da maior area para menor area
    listc = sorted(listc,key = lambda re: -re[2]*re[3])
    for riten in listc:
        if riten in listarem:
            continue
        temr = False
        for srit in lists:
            if srit.contains(riten):
                listarem.append(riten)
                temr = True
        if not temr:
            lists.append(SuperRect(riten,conta))
            conta += 1
            listarem.append(riten)
    return lists

#retorna a lista de retangulos aglutinados em duas etapas
def detectMov(img):
    ret,thresh = cv2.threshold(img,127,255,0)
    #para operar no opencv 2.4
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #para operar no opencv 3.1    
    #im2,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)<1 :
       return []
   #[(0,0,1,1)],[SuperRect((0,0,1,1),-1)]
    listc = []
    for cnt in  contours:
        x,y,w,h=cv2.boundingRect(cnt)
        px = w/2.5
        py = h/2.5
        x =  int(x - px)
        y =  int(y - py)
        w += int(2*px)
        h += int(2*py)
        if w*h>25:
            listc.append((x,y,w,h))
    lists = aglutina(listc)
    listc2 = []
    for sr in lists:
        listc2.append(sr.getRect())
    lists2 = aglutina(listc2)
    #area de restricao dos retangulos
    (hr, wr) = img.shape[:2]
    rectpts = []
    for reci in lists2:
        rectpts.append(reci.getAjRect(wr,hr))

    return rectpts

if __name__=="__main__":
    print args.video

    cap = cv2.VideoCapture(args.video)
    parts =  args.video.split('-')
    npart = parts[1].split('.')
    caminho = os.path.join("/iscsi","videos","sequence","p.{}".format(npart[0]))
    if not os.path.exists(caminho):
        os.makedirs(caminho)
        print "Create {}".format(caminho)
    print cap
    fatorg = 0.125
    conta = 0
    t = clock()
    fgbg = cv2.BackgroundSubtractorMOG()
    while(True):    
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            #cv2.imwrite('frame_{:04d}.jpg'.format(conta),frame)
            reduzido = cv2.resize(frame,(0,0),fx=fatorg,fy=fatorg)
            fgmask = fgbg.apply(reduzido,learningRate=0.001)
            rectpts = detectMov(fgmask)
            grava = False
            for x0,y0,x1,y1 in rectpts:
                x0 = int(x0)
                y0 = int(y0)
                x1 = int(x1)
                y1 = int(y1)
                if (x1-x0)/fatorg > 100 or (y1-y0)/fatorg > 100:
                    grava = True
            if grava:
                arquivo = os.path.join(caminho,"frame-{:07d}.jpg".format(conta))
                print arquivo
                cv2.imwrite(arquivo,frame)
            if conta % 2000 == 0 and conta >0:
                dt = clock() - t
                fps = 2000.0/dt
                #cv2.imshow("Resultado",frame)
                print "Time {} s e {} fps".format(dt,fps)
                t = clock()
            conta += 1
            #if conta > 20000:
            #    break
    cap.release()

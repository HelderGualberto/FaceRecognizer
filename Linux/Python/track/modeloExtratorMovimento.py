#!/usr/bin/env python2
# Autor: Roberto Kenji Hiramatsu
#
# Copyright 2016 Universidade de Sao Paulo
#
# Processamento seletivo e reducao de ocorrencia de deteccao faces fp
#
# Data inicial: 2016-11-29 inicial

import cv2
#import cv2.cv as cv
import numpy as np
from cmath import rect

#cap = cv2.VideoCapture('rtsp://admin:B30cd4Ro@192.168.10.180:554/LiveMedia/ch1/Media2')
cap = cv2.VideoCapture('rtsp://admin:B30cd4Ro@192.168.10.180:554/LiveMedia/ch1/Media1')
#cap = cv2.VideoCapture('/home/yakumo/teste.mp4')
#cap = cv2.VideoCapture('D:\\projetos\\Safety_City_offdata\\teste\\teste.mp4')
#cap = cv2.VideoCapture(0)

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

def aglutina(listc):
    lists = []
    #listarem = []
    conta = 0 
    listc = sorted(listc,key = lambda re: -re[2]*re[3])
    for riten in listc:
        #if riten in listarem:
        #    continue
        temr = False
        for srit in lists:
            if srit.contains(riten):
                #listarem.append(riten)
                temr = True
        if not temr:
            lists.append(SuperRect(riten,conta))
            conta += 1
            #listarem.append(riten)
    return lists

def detectMov(img):
    #somaf = cv2.integral(img)
    #h,w = img.shape
    ret,thresh = cv2.threshold(img,127,255,0)
    #contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    im2,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)<1 :
       return [(0,0,1,1)],[SuperRect((0,0,1,1),-1)]
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
    return listc2,lists2
#opencv 2.4
#fgbg = cv2.BackgroundSubtractorMOG()
#opencv 3.1
#fgbg = cv2.createBackgroundSubtractorMOG2()
#opencv 3.1 GMG
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg   = cv2.createBackgroundSubtractorKNN()

while(1):
    ret, frame = cap.read()
    if ret:
        #fgmask = fgbg.apply(frame,learningRate=0.002)
        frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
        (h, w) = frame.shape[:2]
        Mtr = np.float32([[0,1,0],[-1,0,w]])
        frame = cv2.warpAffine(frame, Mtr, (h,w))        
        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel)
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        listc,lists = detectMov(fgmask)
        cv2.imshow('mask',fgmask)
        for x,y,w,h in listc:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        for sr      in lists:
            x,y,w,h = sr.getRect()
            if w*h < 100 :
                continue
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,str(sr.idx),(x+w/2,y+h/2),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),1)
        cv2.imshow('frame',frame)
        #if len(lists)>2:
        #    if cv2.waitKey(0) & 0xff == 27:
        #       break
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

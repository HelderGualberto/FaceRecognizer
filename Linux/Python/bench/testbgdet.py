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
from common import clock, draw_str

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str,
                    help="url do video",
                    #default='D:\\temp\\teste\\corredor-teste-20161221.mp4')
                    default='D:\\temp\\teste\\corredor-1222040232.mp4')
                    #default='rtsp://admin:B30cd4Ro@127.0.0.1:8554/LiveMedia/ch1/Media1')
                    #default='http://192.168.10.236:7676/repository_servlet/5fg5te4vpsrip7ajelfeuvgpqp')

args = parser.parse_args()
print args.video

cap = cv2.VideoCapture(args.video)
#0)
print cap
conta = 0
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg   = cv2.createBackgroundSubtractorKNN()
t = clock()
while(True):    
    # Capture frame-by-frame
    ret, frame = cap.read()
    #print ret
    # Our operations on the frame come here
    

    # Display the resulting frame
    if ret:
        reduzido = cv2.resize(frame,(0,0),fx=0.125,fy=0.125)
        fgmask = fgbg.apply(reduzido)
        fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel)
        cv2.imshow("Mask",fgmask)
        #gray = cv2.cvtColor(reduzido, cv2.COLOR_BGR2GRAY)            
        if conta % 500 == 0 and conta >0:            
            dt = clock() - t
            fps = 500.0 / dt
            print "Time {} s e {} fps".format(dt,fps)
            #cv2.imshow("video",gray)
                       
            t = clock()
        conta += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        

# When everything done, release the capture
cap.release()


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
                    default='../corredor/corredor-1223084200.mp4')
                    #default='rtsp://admin:B30cd4Ro@127.0.0.1:8554/LiveMedia/ch1/Media2')
                    #default='rtsp://admin:B30cd4Ro@127.0.0.1:8554/LiveMedia/ch1/Media1')

args = parser.parse_args()
print args.video

cap = cv2.VideoCapture(args.video)
#0)
print cap
conta = 0
t = clock()
fgbg = cv2.BackgroundSubtractorMOG()
while(True):    
    # Capture frame-by-frame
    ret, frame = cap.read()
    #print ret
    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    if ret:
        #cv2.imwrite('frame_{:04d}.jpg'.format(conta),frame)
        reduzido = cv2.resize(frame,(0,0),fx=0.2,fy=0.2)
        fgmask = fgbg.apply(reduzido,learningRate=0.001)
        if conta % 500 == 0 and conta >0:
            dt = clock() - t
            fps = 500.0/dt
            #cv2.imshow("Resultado",frame)
            print "Time {} s e {} fps".format(dt,fps)
            t = clock()
        conta += 1
        if conta > 4000:
            break
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
    

# When everything done, release the capture
cap.release()

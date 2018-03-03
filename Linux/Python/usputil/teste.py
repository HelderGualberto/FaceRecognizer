#!/usr/bin/env python2
# Autor: Roberto Kenji Hiramatsu
#
# Copyright 2016 Universidade de Sao Paulo
#
# Modelo para rastrear face para uso de classificacao sobre a camera
#
# Data inicial: 2016-11-09
#             

import argparse

from common import clock
import cv2

#import RepUtil

import numpy as np
import math

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str,
                    help="Video a capturar",
                    #default='/home/yakumo/HBPVR')
                    default='rtsp://admin:B30cd4Ro@192.168.10.180:554/LiveMedia/ch1/Media2') #D:\\temp


if __name__=="__main__":
    args = parser.parse_args()
    print "Carregando video em {}".format(args.video)

    cap = cv2.VideoCapture(args.video)
    temframe=True
    conta = 0
    while(temframe):
            t = clock()
            temframe, frame = cap.read()
            if temframe:
                #teste de rotacao
                (h, w) = frame.shape[:2]
                #center = (w / 2, h / 2)
                center = (0 , h)
                # rotate the image by 180 degrees
                #M = cv2.getRotationMatrix2D(center, -90, 1.0)
                #rotated = cv2.warpAffine(frame, M, (h, w+h))
                Mtr = np.float32([[0,-1,h],[1,0,0]])
                rotated = cv2.warpAffine(frame, Mtr, (h,w))
                cv2.imshow("rotated", rotated)
                cv2.imshow("normal",frame)

                cv2.waitKey(1)
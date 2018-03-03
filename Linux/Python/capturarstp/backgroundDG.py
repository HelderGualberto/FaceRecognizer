# Autor: Roberto Kenji Hiramatsu
# Teste local para verificar capacidade ocupada de processamento em maquina local e uso de modo de tracking para otimizar captura dos dados
# Data: 2016-09-05


import numpy as np
import cv2
from imutils import paths
import argparse
from common import clock, draw_str
from datetime import datetime
import thread
import copy
import pymongo
import gridfs
from pymongo import MongoClient
import gridfs
import os

client = MongoClient("mongodb://nuvemusp.pad.lsi.usp.br:37027")
db = client.gridfs

fs = gridfs.GridFS(db)

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str,
                    help="url do video",
                    default='rtsp://kenji:6qi7j94i@127.0.0.1:8554/H264')                    
#                    default='rtsp://admin:B30cd4Ro@192.168.10.180:554/LiveMedia/ch1/Media1')

parser.add_argument('--cascade', type=str,
                    help="cascade haar detector",
                    default='haarcascades/haarcascade_frontalface_alt.xml')

args = parser.parse_args()
print args.video





def detect(img, cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2,
                                     minSize=(50, 50),
                                     flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return [],0.0
    rects[:,2:] += rects[:,:2]
    bdisp = 0.0
    if len(rects)>0:
        x1, y1, x2, y2 = rects[0]
        bdisp  =  cv2.Laplacian(gray[y1:y2,x1:x2],cv2.CV_64F).var()
    return rects,bdisp

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

#cv2.ocl.setUseOpenCL(False)
def detectaM(imag,sx,sy):
    somaf = cv2.integral(imag)
    h,w = imag.shape
    px = w / sx
    py = h / sy
    sqv = []
    bmax = float(w*h/(sx*sy))*255.0
    resp = False
    xmin = w
    ymin = h
    xmax = 0
    ymax = 0
    for ix in range(0,sx):
        for iy in range(0,sy):
            x1 = ix * px
            y1 = iy * py
            x2 = (ix + 1) * px 
            y2 = (iy + 1) * py 
            v = somaf[y2,x2]-somaf[y2,x1]-somaf[y1,x2]+somaf[y1,x1]
            sqv.append(v)
            pp = float(v)/bmax
            #print ix,':',iy,'->',v," per:",pp
            if pp > 0.1:
                #print ix,':',iy,'(',bmax,')->',v," per:",pp
                xmin = x1 if x1 < xmin else xmin
                ymin = y1 if y1 < ymin else ymin
                xmax = x2 if x2 > xmax else xmax
                ymax = y2 if y2 > ymax else ymax
                #return True
                resp = True
            #    cv2.imshow('blackmask',imag)
    if resp:
       xmin = xmin - px if xmin >= px else xmin
       ymin = ymin - py if ymin >= py else ymin
       xmax = xmax + px if (xmax + px) <= w else xmax
       ymax = ymax + py if (ymax + py) <= h else ymax
       
       print xmin,ymin,xmax,ymax
    return resp,np.array([xmin,ymin,xmax,ymax])
            #    cv2.waitKey(0)

def gravaIm(img,arquivo,mesdata,horamin,local,frmc,bfram,areaxy):
    cv2.imwrite(arquivo,img)
    fileId = fs.put(open(arquivo,'rb'),filename=arquivo,localidade=local,mesdia=mesdata,horaminuto=horamin,frame=frmc,
                    framebase=bfram,area=(areaxy[0],areaxy[1],areaxy[2],areaxy[3]))
    print "Enviado arquivo",arquivo
    os.remove(arquivo)

if __name__ == '__main__':
  try:
    cascade = cv2.CascadeClassifier(args.cascade)
    cap = cv2.VideoCapture(args.video)
    print cap
    fgbg  = cv2.BackgroundSubtractorMOG()
    #(history=100,varThreshold=16)
    #(100,3,0.3)
    conta = 0
    baseframe = True 
    while(True):
        t = clock()
        ret, frame = cap.read()
        #dt = clock() - t
        #print 'C Time',dt,' s'  
        if ret:
            conta += 1
            reduzido = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
            #if conta < 2:
            #    [h,w,p] = reduzido.shape
            #    fgm  = np.zeros((h,w,p),np.uint8)
            fgm = fgbg.apply(reduzido,learningRate=0.001)
            #print reduzido.shape,fgm.shape 
            #thread.start_new_thread(gravaIm,(frame.copy(),"frame{:03d}.jpg".format(conta)))
            resp,area = detectaM(fgm,16,6)       
            if resp:
                now = datetime.now()
                mesdata = now.month*100 + now.day
                horamin = now.hour*100 + now.minute
                
                arquivo='frame_{:02d}.{:02d}.{:02d}.{:02d}.{:02d}_f{:09d}.jpg'.format(now.month,now.day,                                                                                                   now.hour,now.minute,
                                                                                                   now.second,conta)
                if baseframe:
                    thread.start_new_thread(gravaIm,(copy.copy(frame),copy.copy(arquivo),mesdata,horamin,"corredor",conta,baseframe,copy.copy(2*area)))
                    baseframe = False
                else:
                    (x1,y1,x2,y2) = 2 * area
                    thread.start_new_thread(gravaIm,(copy.copy(frame[y1:y2,x1:x2]),copy.copy(arquivo),mesdata,horamin,"corredor",conta,baseframe,copy.copy(2*area)))
                dt = clock() - t
                print 'C Time',dt,' s em ',now
            else:
                baseframe = True
                #thread.start_new_thread(gravaIm,(fgm.copy(),copy.copy("bg"+arquivo)))       
            #print fgmask.shape
            #thread.start_new_thread(gravaIm,(fgm.copy(),"back{:03d}.jpg".format(conta))) 
            #cv2.imshow('frame',frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #        break
  except KeyboardInterrupt:
    # When everything done, release the capture
    #cv2.destroyAllWindows()
    cap.release()

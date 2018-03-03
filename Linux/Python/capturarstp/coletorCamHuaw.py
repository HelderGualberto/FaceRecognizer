# Autor: Roberto Kenji Hiramatsu
# Teste local para verificar capacidade ocupada de processamento em maquina local e uso de modo de tracking para otimizar captura dos dados
# Data: 2016-09-05


import numpy as np
import cv2
from imutils import paths
import argparse
from common import clock, draw_str
from datetime import datetime
from pymongo import MongoClient
import gridfs
import os
import thread
from io import BytesIO
#import StringIO
#import binascii

#===============================================================================
# import StringIO
# import matplotlib.pyplot as plt
# import urllib
# import base64
#===============================================================================

client = MongoClient("mongodb://videobroker.pad.lsi.usp.br:37027")
db = client.gridfs

fs = gridfs.GridFS(db)


parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str,
                    help="url do video",
                    #default='rtsp://kenji:6qi7j94i@192.168.10.181:554/H264')                    
                    default='rtsp://admin:B30cd4Ro@192.168.10.180:554/LiveMedia/ch1/Media1')

parser.add_argument('--cascade', type=str,
                    help="cascade haar detector",
                    default='haarcascades/haarcascade_frontalface_alt.xml')

args = parser.parse_args()
print args.video

def detect(img, cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2,
                                     minSize=(30, 30),
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
       #print xmin,ymin,xmax,ymax
    return resp,np.array([xmin,ymin,xmax,ymax])

def novoEquad(areaequa,wib,heb):
    x1,y1,x2,y2 = areaequa
    dx = (x2-x1)/6
    x1 = (x1 - dx) if (x1 - dx)>0 else 0
    x2 = (x2 + dx) if (x2 + dx)<wib else wib
    dy = (y2 -y1)/6
    y1 = (y1 - dy) if (y1 - dy)>0 else 0
    y2 = (y2 + 2*dy) if (y2 + 2*dy) < heb else heb
    return x1,y1,x2,y2

def gravaIm(img,arquivo,mesdata,horamin,local,frmc,bfram,areaxy):
    #cv2.imwrite(arquivo,img)
    #fileId = fs.put(open(arquivo,'rb'),filename=arquivo,localidade=local,mesdia=mesdata,horaminuto=horamin,frame=frmc,
    #                framebase=bfram,area=(areaxy[0],areaxy[1],areaxy[2],areaxy[3]))
    buf = cv2.imencode(".jpg",img)
    #imgio = StringIO.StringIO()
    imgio = BytesIO()
    imgio.write(buf[1])
    imgio.seek(0)
    try:
        fgrid = fs.new_file(filename=arquivo,localidade=local,mesdia=mesdata,horaminuto=horamin,frame=frmc,
                    framebase=bfram,area=(areaxy[0],areaxy[1],areaxy[2],areaxy[3]),tratado=False)
        fgrid.write(imgio)
    finally:
        fgrid.close()
    imgio.close()
    print "Enviado arquivo",arquivo
    #os.remove(arquivo)

if __name__ == '__main__':
    try:
        cascade = cv2.CascadeClassifier(args.cascade)
        cap = cv2.VideoCapture(args.video)
        print cap
        fgbg = cv2.BackgroundSubtractorMOG()
        conta = 0
        while(True):
            t = clock()
            # Capture frame-by-frame
            ret, frame = cap.read()
            #print ret
            # Our operations on the frame come here
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #dt = clock() - t
            #print 'C Time',dt,' s'  
            # Display the resulting frame
            if ret:
                #print frame.shape
                reduzido = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
                #print reduzido.shape
                fgmask = fgbg.apply(reduzido,learningRate=0.001)
                resp,area = detectaM(fgmask,12,6)
                #cv2.imshow("MapaB",fgmask)
                (xr1,yr1,xr2,yr2) = area
                if resp:
                    draw_rects(reduzido, [area], (0,255,0))
                    #cv2.imshow("Movimento",reduzido)                
                    rects,bdisp=detect(reduzido[yr1:yr2,xr1:xr2], cascade)
                    contaf = 0
                    for rectface in rects:
                        #and bdisp > 200.0
                        contaf += 1
                        areadeiteresse = np.array(rectface)
                        arearecal = areadeiteresse + np.array([xr1,yr1,xr1,yr1]) 
                        draw_rects(reduzido, [arearecal] , (0, 0, 255))
                        print "Bidx:",bdisp
                        arearecal = 4 * arearecal
                        hf,wf,pf = frame.shape
                        xf1,yf1,xf2,yf2 = novoEquad(arearecal,wf,hf)
                    
                        now = datetime.now()
                        medi = now.month*100 + now.day
                        homi = now.hour*100 + now.minute
                
                        if (xf2-xf1)< 600:
                            arq='face_{:02d}.{:02d}.{:02d}.{:02d}.{:02d}_f{:09d}_{}.jpg'.format(now.month,now.day,                                                                                                   now.hour,now.minute,
                                                                                                   now.second,conta,contaf)                    
                            #cv2.imwrite(arquivo,frame[yf1:yf2,xf1:xf2])
                            #                                          img,arquivo,mesdata,horamin,local,frmc,bfram,areaxy
                            thread.start_new_thread(gravaIm,(frame[yf1:yf2,xf1:xf2].copy(), arq ,medi,
                                                                   homi,"HumanLab",conta,False,(xf1,yf1,xf2,yf2)))
                        dt = clock() - t
                        print 'D Time',dt,' s' 
                        #(x1,y1,x2,y2) = 4 * np.array(rects)
                    #if len(rects)>0:
                    #    cv2.imwrite("frame{:04d}.jpg".format(conta),reduzido)
                        #cv2.imshow("Corredor",frame)
                #dt = clock() - t
                #print 'S Time',dt,' s'
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break
            conta += 1
    except KeyboardInterrupt:
        # When everything done, release the capture
        cap.release()

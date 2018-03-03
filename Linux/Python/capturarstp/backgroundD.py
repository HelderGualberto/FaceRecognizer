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

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str,
                    help="url do video",
                    default='rtsp://kenji:6qi7j94i@192.168.10.181:554/H264')                    
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

cv2.ocl.setUseOpenCL(False)
def detectaM(imag,sx,sy):
    somaf = cv2.integral(imag)
    h,w = imag.shape
    px = w / sx
    py = h / sy
    sqv = []
    bmax = float(w*h/(sx*sy))*255.0
    for ix in range(0,sx):
        for iy in range(0,sy):
            x1 = ix * px
            y1 = iy * py
            x2 = (ix + 1) * px -1
            y2 = (iy + 1) * py -1
            v = somaf[y2,x2]-somaf[y2,x1]-somaf[y1,x2]+somaf[y1,x1]
            sqv.append(v)
            #print ix,':',iy,'->',v
            pp = float(v)/bmax
            if pp > 0.1:
                return True
            #    cv2.imshow('blackmask',imag)
        return False
            #    cv2.waitKey(0)
def gravaIm(img,arquivo):
    cv2.imwrite(arquivo,frame)

if __name__ == '__main__':
    cascade = cv2.CascadeClassifier(args.cascade)
    cap = cv2.VideoCapture(args.video)
    print cap
    fgbg = cv2.createBackgroundSubtractorMOG2()
    conta = 0 
    while(True):
        t = clock()
        ret, frame = cap.read()
        dt = clock() - t
        print 'C Time',dt,' s'  
        if ret:
            conta += 1
            reduzido = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
            fgmask = fgbg.apply(reduzido)
            if detectaM(fgmask,2,2):
                now = datetime.now()
                arquivo='frame_{:02d}.{:02d}.{:02d}.{:02d}.{:02d}_f{:08d}.jpg'.format(now.month,now.day,                                                                                                   now.hour,now.minute,
                                                                                                   now.second,conta)
                thread.start_new_thread(gravaIm,(frame.copy(),copy.copy(arquivo)))       
            #print fgmask.shape
            #cv2.imshow('frame',frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #        break

# When everything done, release the capture
cv2.destroyAllWindows()
cap.release()

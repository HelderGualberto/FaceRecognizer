# Autor: Roberto Kenji Hiramatsu
# Teste local para verificar capacidade ocupada de processamento em maquina local e uso de modo de tracking para otimizar captura dos dados
# Data: 2016-09-05


import numpy as np
import cv2
from imutils import paths
import argparse
from common import clock, draw_str

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


if __name__ == '__main__':
    cascade = cv2.CascadeClassifier(args.cascade)
    cap = cv2.VideoCapture(args.video)
    print cap
    conta = 0
    while(True):
        t = clock()
        # Capture frame-by-frame
        ret, frame = cap.read()
        #print ret
        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dt = clock() - t
        print 'C Time',dt,' s'  
        # Display the resulting frame
        if ret:
            #reduzido = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
            #reduzido        
            rects,bdisp=detect(frame, cascade)
            if len(rects)>0 and bdisp > 200.0:
                draw_rects(frame, rects, (0, 255, 0))
                print "Bidx:",bdisp
            #cv2.imwrite('frame_{:04d}.jpg'.format(conta),frame)
            dt = clock() - t
            print 'D Time',dt,' s'        
            cv2.imshow("Corredor",frame)
            dt = clock() - t
            print 'S Time',dt,' s'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        conta += 1

# When everything done, release the capture
cap.release()

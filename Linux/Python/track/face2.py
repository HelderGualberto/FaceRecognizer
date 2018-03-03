
import argparse
import cv2
import numpy as np

from common import clock

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str,
                    help="Video a capturar",
                    #default='rtsp://kenji:6qi7j94i@192.168.10.181:554/H264') 
                    #default='rtsp://admin:B30cd4Ro@192.168.10.180:554/LiveMedia/ch1/Media1') 
                    #default='/home/yakumo/HBPVR')
                    default=0) #D:\\temp

#alt2 apresenta melhor desempenho para detecao de face
parser.add_argument('--cascade', type=str,
                    help="cascade haar detector",
                    #default='haarcascades/haarcascade_frontalface_alt_tree.xml')
                    #default='haarcascades/haarcascade_frontalface_alt2.xml')
                    default='haarcascades\\haarcascade_frontalface_alt2.xml') #D:\\app\\opencv31\\sources\\data\\
                    #default='haarcascades/haarcascade_frontalface_default.xml')
                    #default='haarcascades/haarcascade_frontalface_alt.xml')
                    #default='D:\\app\\opencv31\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml')
                    
if __name__=="__main__":
    args = parser.parse_args()
    print "Carregando video em {}".format(args.video)
    #cascade = cv2.CascadeClassifier(args.cascade)

    # objeto para controle de rastreamento de faces
    cap = cv2.VideoCapture(args.video)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    temframe = True
    while temframe:
            t = clock()
            temframe, frame = cap.read()
            if temframe:
                anotado = frame.copy()
                reduzido = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
                fgmask = fgbg.apply(reduzido)
                cv2.imshow("Anotado",anotado)
                cv2.imshow("Mask",fgmask)
                dt = clock() - t
                print "Total {} ms".format(dt*1000)
                # apresentado amostragem
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    # When everything done, release the capture
    cap.release()

import numpy as np
import cv2
from imutils import paths
import argparse
from common import clock, draw_str

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str,
                    help="url do video",
                    default='rtsp://admin:B30cd4Ro@192.168.10.180:554/LiveMedia/ch1/Media1')

args = parser.parse_args()
print args.video

cap = cv2.VideoCapture(args.video)
#0)
print cap
conta = 0
while(True):
    t = clock()
    # Capture frame-by-frame
    ret, frame = cap.read()
    #print ret
    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    if ret:
        #cv2.imwrite('frame_{:04d}.jpg'.format(conta),frame)
        reduzido = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
        cv2.imshow("Huaweicam",reduzido)
        dt = clock() - t
        print 'Time',dt,' s'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    conta += 1

# When everything done, release the capture
cap.release()

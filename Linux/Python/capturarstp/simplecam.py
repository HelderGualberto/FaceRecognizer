import numpy as np
import cv2
#rtsp://admin:B30cd4Ro@192.168.20.194:554/LiveMedia/ch1/Media1
cap = cv2.VideoCapture('rtsp://admin:B30cd4Ro@127.0.0.1:8554/video')
#0)
print cap

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    print ret
    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    if ret:        
        cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
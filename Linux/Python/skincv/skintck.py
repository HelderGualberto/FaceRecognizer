# import the necessary packages
#from pyimagesearch 
import imutils
import numpy as np
import argparse
import cv2
#para plotar imagem do histograma
from matplotlib import pyplot as plt
import copy 
 
 
def aceitaOverLap(q1,q2,fator):
    (x1,y1,w1,h1)=q1
    (x2,y2,w2,h2)=q2
    x1 -= w1/10
    y1 -= h1/10
    w1 += w1/5
    h1 += h1/5
    x2 -= w2/10
    y2 -= h2/10
    w2 += w2/5
    h2 += h2/5
    #determina o maior  x minimo
    xmi = x1 if x1 > x2 else x2
    #determina o maior  y minimo
    ymi = y1 if y1 > y2 else y2
    #determina o menor x maximo
    xma = x1 + w1 if (x1 + w1) < (x2 + w2) else (x2 + w2)
    #determina o menor y maximo
    yma = y1 + h1 if (y1 + h1) < (y2 + h2) else (y2 + h2)
    #area da interseccao 
    areai = (xma-xmi)*(yma-ymi)
    areaa= w2 * h2
    raz=float(areai)/float(areaa)
    print 'raz ',raz,' areai ',areai,' areaa ',areaa
    if raz > fator:
        return True
    return False

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help = "path to the (optional) video file")
args = vars(ap.parse_args())
#carregando cascasde de rosto
face_cascade = cv2.CascadeClassifier('D:\projetos\Safety_City\Code\data\haarcascades\haarcascade_frontalface_default.xml')
# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")
# if a video path was not supplied, grab the reference
# to the gray
if not args.get("video", False):
	camera = cv2.VideoCapture(0)
 
# otherwise, load the video
else:
	camera = cv2.VideoCapture(args["video"])
        trackproc = False
        trackproc2 = False
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )        
	# keep looping over the frames in the video
        idxfrm = 1        
	while True:
		# grab the current frame
		(grabbed, frame) = camera.read()
 		facecolor = [(255,0,0),(0,255,0),(0,0,255),(128,0,0),(0,128,0),(0,0,128)]
                
		# if we are viewing a video and we did not grab a
		# frame, then we have reached the end of the video
		if args.get("video") and not grabbed:
			break
                # processamento de elemento track
                if trackproc2 :
                    print('Processa track')
                    hf , wf = frame.shape[:2]
                    idxh = 0
                    for (x,y,w,h) in facesAnt2:
                        # selecao para procurar nesse frame
                        roi_procu = copy.copy(frame[y-h/2 if (y-h/2)>0 else 1: y + 3*h/2 if (y + 3*h/2) < hf else hf,
                        x-w/2 if (x-w/2)>0 else 1 :x+3*w/2 if (x+3*w/2) < wf else wf] )                      
                        nhf , nwf = roi_procu.shape[:2]
                        cv2.imshow('Procurando na imagem track',roi_procu)
                        roi_hsv = cv2.cvtColor(roi_procu, cv2.COLOR_BGR2HSV)
                        dstTrack = cv2.calcBackProject([roi_hsv],[0],roihistAnt2[idxh],[0,180],1)
                        # apply meanshift to get the new location
                        xa,ya,wa,ha =(w/2 if (x-w/2)>0 else x,h/2 if(y-h/2)>0 else y ,w,h)
                        ret, track_window = cv2.CamShift(dstTrack, (xa,ha,wa,ha), term_crit)
                        print 'Tracking window 2',track_window,' de ',(xa,ha,wa,ha)
                        # Draw it on image
                        #pts = cv2.boxPoints(ret)
                        #pts = np.int0(pts)
                        #img_track = cv2.polylines(roi_procu,[pts],True, 255,2)
                        xn,yn,wn,hn=track_window
                        if aceitaOverLap((xn,yn,wn,hn),(xa,ya,wa,ha),0.5):
                            cv2.rectangle(roi_procu,(xa,ya),(xa+wa,ya+ha),(255,255,255),2) 
                            cv2.rectangle(roi_procu,(xn,yn),(xn+wn,yn+hn),facecolor[idxh % 6],2) 
                            textomsg = 'Frame '+str(idxfrm)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(roi_procu,textomsg,(0,30),font,0.5,(255,255,255),2)                            
                            cv2.imshow('Trackeado 2',roi_procu)
                            if cv2.waitKey(0) & 0xFF == ord("q"):
                                break 
                        idxh += 1

                        
                trackproc2 = False        
                if trackproc :
                    print('Processa track')
                    hf , wf = frame.shape[:2]
                    idxh = 0
                    print 'Tem hists',len(roihistAnt )
                    facesAnt2 = []
                    roihistAnt2 = []
                    for (x,y,w,h) in facesAnt:
                        # selecao para procurar nesse frame
                        roi_procu = copy.copy(frame[y-h/2 if (y-h/2)>0 else 1: y + 3*h/2 if (y + 3*h/2) < hf else hf,
                        x-w/2 if (x-w/2)>0 else 1 :x+3*w/2 if (x+3*w/2) < wf else wf] )                      
                        nhf , nwf = roi_procu.shape[:2]
                        cv2.imshow('Procurando na imagem',roi_procu)
                        roi_hsv = cv2.cvtColor(roi_procu, cv2.COLOR_BGR2HSV)
                        dstTrack = cv2.calcBackProject([roi_hsv],[0],roihistAnt[idxh],[0,180],1)
                        # apply meanshift to get the new location
                        xa,ya,wa,ha =(w/2 if (x-w/2)>0 else x,h/2 if(y-h/2)>0 else y ,w,h)
                        ret, track_window = cv2.CamShift(dstTrack, (xa,ha,wa,ha), term_crit)
                        print 'Tracking window ',track_window,' de ',(xa,ha,wa,ha)
                        # Draw it on image
                        #pts = cv2.boxPoints(ret)
                        #pts = np.int0(pts)
                        #img_track = cv2.polylines(roi_procu,[pts],True, 255,2)
                        xn,yn,wn,hn=track_window
                        if aceitaOverLap((xn,yn,wn,hn),(xa,ya,wa,ha),0.5):
                            facesAnt2.append(((x-w/2 if (x-w/2)>0 else 1)+xn,(y-h/2 if (y-h/2)>0 else 1)+yn,wn,hn))
                            roihistAnt2.append(roihistAnt[idxh])
                            trackproc2 = True
                            cv2.rectangle(roi_procu,(xa,ya),(xa+wa,ya+ha),(255,255,255),2) 
                            cv2.rectangle(roi_procu,(xn,yn),(xn+wn,yn+hn),facecolor[idxh % 6],2)
                            textomsg = 'Frame '+str(idxfrm)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(roi_procu,textomsg,(0,30),font,0.5,(255,255,255),2)                            
                            cv2.imshow('Trackeado',roi_procu)
                            if cv2.waitKey(0) & 0xFF == ord("q"):
                                break 
                        idxh += 1
 
 		# obtendo a imagem em escala de cinza
 		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 		#detectando as faces na imagem
 		faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(120, 120),flags=cv2.CASCADE_SCALE_IMAGE)
 		img_sw = copy.copy(frame)
 		idxc = 0
                roihistAnt = []
                #if not faces.size:
                trackproc  = False  
 		for (x,y,w,h) in faces:
 			# recorte da area de interesse da imagem a 20% do topo 50% da base e 33% de cada lado da area de face localizada
 			#roi_img = frame[(y+(h/5)):y+h-h/2, (x+w/3):x+w-(w/3)]
                        # recorte da area de interesse da imagem a 30% do topo 40% da base e 33% de cada lado da area de face localizada
                        roi_img = frame[(y+(3*h/10)):y+h-2*h/5, (x+w/3):x+w-(w/3)]
 			cv2.rectangle(img_sw,(x,y),(x+w,y+h),facecolor[idxc % 6],2) 			
 			#
 			#histograma rgb
 			#color = ('b','g','r')
 			#color = ('h','s','v')
 			#roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
 			#for i,col in enumerate(color):
 			#	histr = cv2.calcHist([roi_img],[i],None,[256],[0,256])
 			#	plt.plot(histr,color = col)
 			#	plt.xlim([0,256])
 			#plt.show()
 			#cv2.imshow('face',roi_img)
 			#cv2.waitKey(0)
 			hsv = cv2.cvtColor(roi_img,cv2.COLOR_BGR2HSV)
 			target = copy.copy(frame)
 			hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
 			# calculating object histogram
 			roihist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
                        mask = cv2.inRange(hsv, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
                        roi_hist_tk = cv2.calcHist([hsv],[0],mask,[180],[0,180])
                        roihistAnt.append(roi_hist_tk)
 			# normalize histogram and apply backprojection
 			cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
 			dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
 			# Now convolute with circular disc
 			disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
 			cv2.filter2D(dst,-1,disc,dst)
 			# threshold and binary AND
 			ret,thresh = cv2.threshold(dst,50,255,0)
 			thresh = cv2.merge((thresh,thresh,thresh))
 			res = cv2.bitwise_and(target,thresh)
 			cv2.imshow('mascara',thresh)
 			cv2.rectangle(res,(x,y),(x+w,y+h),facecolor[idxc % 6],2)
 			idxc += 1
 			cv2.imshow('Resultado',res)
                        # salva o valor da ultima rodada 
                        trackproc  = True  
 			#if cv2.waitKey(1000) & 0xFF == ord("q"):
                        #    break
                textomsg = 'Frame '+str(idxfrm)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img_sw,textomsg,(0,30),font,0.5,(255,255,255),2)
 		cv2.imshow('faces detectadas',img_sw)
 		
                if trackproc:
                    facesAnt = faces                              
		# resize the frame, convert it to the HSV color space,
		# and determine the HSV pixel intensities that fall into
		# the speicifed upper and lower boundaries
		
		#frame = imutils.resize(frame, width = 400)
		#converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		#skinMask = cv2.inRange(converted, lower, upper)
 
		# apply a series of erosions and dilations to the mask
		# using an elliptical kernel
		#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
		#skinMask = cv2.erode(skinMask, kernel, iterations = 2)
		#skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
 
		# blur the mask to help remove noise, then apply the
		# mask to the frame
		#skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
		#skin = cv2.bitwise_and(frame, frame, mask = skinMask)
 
		# show the skin in the image along with the mask
		#cv2.imshow("images", np.hstack([frame, skin]))
 
		# if the 'q' key is pressed, stop the loop
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break
                idxfrm += 1
 
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

	

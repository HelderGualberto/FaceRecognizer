#!/usr/bin/env python

import cv2
import numpy as np
from align_dlib import AlignDlib

from imutils import paths

import os
from os.path import join
import math
#solvePnPHPAng,
from pose import calcHVAngRosto
from numpy.linalg import inv
# Read Image
#im = cv2.imread("head.png");

#2D image points. If you change the image, you need to change vector
#===============================================================================
# image_points = np.array([
#                             (359, 391),     # Nose tip
#                             (399, 561),     # Chin
#                             (337, 297),     # Left eye left corner
#                             (513, 301),     # Right eye right corne
#                             (345, 465),     # Left Mouth corner
#                             (453, 469)      # Right mouth corner
#                         ], dtype="double")
#===============================================================================

#pathimg='D:\\Safety_City\\imagens\\base00'
#pathimg='E:\\arquivo\\temp\\fotos'
pathimg = "../../../imagens/qualityimg"
#------------------------------------------------------------- #3D model points.
#----------------------------------------------------- model_points = np.array([
                            #----------- (0.0, 0.0, 0.0),             # Nose tip
                            #----------- #(0.0, 179.0, -60.0),        # med eyes
                            #------------ (0.0, 130.0, -60.0),        # med eyes
                            #--------------- (0.0, -330.0, -65.0),        # Chin
                            # (-225.0, 170.0, -135.0),     # Left eye left corner
                            # (225.0, 170.0, -135.0),      # Right eye right corne
                            #-- (-150.0, -150.0, -125.0),    # Left Mouth corner
                            #- (150.0, -150.0, -125.0)      # Right mouth corner
#------------------------------------------------------------------------------ 
                        #---------------------------------------------------- ])
#3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip                           
                            (-940.0, 718.5, -873.5),     # 36 Left eye left corner
                            (940.0, 718.5, -873.5),      # 45 Right eye right corne
                            (0.0, 585.0, -376.0),        # 28 med eyes
                            #(-1500, 207.0, -2058.0),        # 1 canto esquerdo do rosto 
                            #(1500.0, 207.0, -2058.0),    # 15 canto direito do rosto
                            (0.0, -499.0, -193.5),    # 51 canto direito do rosto 
                            (-590.0, 1434.0, -534.0),      # 20 ponto 2 esquerdo
                            (590.0, 1434.0, -534.0)      # 23 ponto 2 direito

                        ])         

                            #===================================================
                            # (-682.0, 1119.0, -534.0),      # 20 ponto 2 esquerdo
                            # (682.0, 1119.0, -534.0)      # 23 ponto 2 direito               
                            #===================================================

align = AlignDlib(join("..","..","data","models","dlib","shape_predictor_68_face_landmarks.dat"))

# do modelo de royshil/HeadPosePnP
#===============================================================================
# model_points = np.array([
#                          (2.37427,110.322,21.7776),     # left center eye
#                          (70.0602,109.898,20.8234),     # right center eye
#                          (36.8301,78.3185,52.0345),     # Nose tip
#                          (14.8498,51.0115,30.2378),     # left  mouth corner
#                          (58.1825,51.0115,29.6224),     # right mouth corner
#                          (-61.8886,127.797,-89.4523),   # left ear
#                          (127.603,126.9,-83.9129)       # right ear
#                          ])
#===============================================================================
def projecaoXYZ(rotationmatrix,translation_vector,camera_matrix,ponto):
    pr = np.matmul(rotationmatrix,ponto.T)
    pt = np.add(pr,translation_vector)
    pp = np.matmul(camera_matrix,pt)
    return pt,pp
    

# Camera internals
def analisaIm(im):
    size = im.shape
    copiado = im.copy()

    focal_length = max(size)

    print "Distancia focal {}".format(focal_length)
    if focal_length < 120:
        try:
            cv2.imshow("ERRO",im)
            cv2.waitKey(0)
        except:
            print "Nao consegue mostrar arquivo!!!"
        return
    cv2.imshow("Input",im)
    #focal_length = size[1]
    center = (size[1]/2, size[0]/2)

    #print "Center:{}".format(center)

    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

    #print "Camera Matrix :\n {0}".format(camera_matrix)


    bb = align.getLargestFaceBoundingBox(im)
    if bb is not None:
        landmarks = align.findLandmarks(im, bb)
    else:
        return

    image_points = np.array([
                             landmarks[30],     # Nose tip
                             landmarks[36],     # Left eye left corner
                             landmarks[45],     # Right eye right corne
                             landmarks[28],     # med eyes low
                             #landmarks[1],     # Canto esquerdo do rosto
                             #landmarks[15],     # canto direito do rosto
                             landmarks[51],     # centro superior da boca
                             landmarks[20],      # ponto 2 esquerdo
                             landmarks[23]      # ponto 2 direito
                         ], dtype="double")

    indexl = [30,36,45,28,51,20,23] #,1,15
    #===========================================================================
    # image_points = np.array([
    #                          landmarks[30],     # Nose tip
    #                          #landmarks[27],     # med eyes
    #                          landmarks[28],     # med eyes low
    #                          landmarks[8],     # Chin
    #                          landmarks[36],     # Left eye left corner
    #                          landmarks[45],     # Right eye right corne
    #                          landmarks[48],     # Left Mouth corner
    #                          landmarks[54]      # Right mouth corner
    #                      ], dtype="double")
    #===========================================================================

#===============================================================================
# image_points = np.array([
#                          (2.37427,110.322,21.7776),     # left center eye
#                          (70.0602,109.898,20.8234),     # right center eye
#                          (36.8301,78.3185,52.0345),     # Nose tip
#                          (14.8498,51.0115,30.2378),     # left  mouth corner
#                          (58.1825,51.0115,29.6224),     # right mouth corner
#                          (-61.8886,127.797,-89.4523),   # left ear
#                          (127.603,126.9,-83.9129)       # right ear
#                          ], dtype="double")
#===============================================================================


    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    #===========================================================================
    #print "Rotation Vector:\n {0}".format(rotation_vector)
    #print "Translation Vector:\n {0}".format(translation_vector)
    rotationmatrix,jacobimat = cv2.Rodrigues(rotation_vector)
    #print "Rotation Matrix:\n {}".format(rotationmatrix)
    # mdsti = np.linalg.inv(mdst)
    # print "Rotation Matrix Inv:\n {}".format(mdsti)
    # 
    # camuvw=np.matmul(mdsti,translation_vector)
    # print "Origin on UVW space(objetct):\n{}".format(camuvw)
    # 
    # (camerapos, jacocam)= cv2.projectPoints(camuvw.T, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    # desv = 100*np.divide(np.subtract(camerapos,center),center)
    # print "Posicao da camera?\n{} com desvio de {}  do centro {}".format(camerapos,desv,center)
    #===========================================================================
# Project a 3D point (0, 0, 1000.0) onto the image plane.
# We use this to draw a line sticking out of the nose

    ponto1000 = np.array([(0.0, 0.0, 1000.0)])
    #ponto rotacionado
    pt1,pp = projecaoXYZ(rotationmatrix,translation_vector,camera_matrix,ponto1000)
    #print "Projecao de {} do espaco de objeto para espaco de camera:\n {} ".format(ponto1000,pt1)    
    #print "No plano da imagem:\n {}".format(pp)
    (nose_end_point2D, jacobian) = cv2.projectPoints(ponto1000, rotation_vector, translation_vector, camera_matrix, dist_coeffs)


    ponto = np.array([(0.0, 0.0, 0.0)])
    #ponto rotacionado
    pt2,pp = projecaoXYZ(rotationmatrix,translation_vector,camera_matrix,ponto)
    #print "Projecao de {} do espaco de objeto para espaco de camera:\n {} ".format(ponto,pt2)    
    #print "No plano da imagem {} contra ponto do nariz em {}".format(pp/pp[2],landmarks[30])
    
    #pontoc = ponto = np.array([(0.0, 0.0, focal_length)])
    #print "R:{},T:{}".format(inv(rotationmatrix),-1.0*translation_vector)
    pr0 = np.matmul(inv(rotationmatrix),-1.0*translation_vector)
    #print "Zero of X in Obj space:{}".format(pr0)
    
    
    vetorp = np.subtract(pt1,pt2)
    #angh = math.atan(vetorp[0][0]/vetorp[2][0])*180/math.pi
    #angv = math.atan(vetorp[1][0]/vetorp[2][0])*180/math.pi
    angh = math.atan(pr0[0][0]/pr0[2][0])*180/math.pi
    angv = math.atan(pr0[1][0]/pr0[2][0])*180/math.pi
    angcab,angvcab,pp=calcHVAngRosto(landmarks)
    print "Vetor formado e:\n {} \n e angulos H:{} V:{} x H:{} V:{}".format(vetorp
                                                                            ,angh,angv
                                                                            ,angcab,angvcab)
    
    (nose_end_point2D, jacobian) = cv2.projectPoints(ponto1000, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    (nosep, jaconose)= cv2.projectPoints(np.array([ (0.0, 0.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    #print "Projecao do vetor {} e ponta do nariz projetada {} contra ponto fornecido {}".format(nose_end_point2D,nosep,landmarks[30])

    #print "Fator escalar s? {} {}".format((pp[0][0]/nose_end_point2D[0][0][0]),(pp[1][0]/nose_end_point2D[0][0][1]))
    
    #ponto obtidos no dlib -> vermelho
    for idx,p in enumerate(image_points):
        cv2.circle(copiado, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
        cv2.putText(copiado,str(indexl[idx]),(int(p[0]), int(p[1])),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,0,255),1)
    #pontos projetados do espaco de objeto para imangem -> verde
    for idx,p in enumerate(model_points):
        (ponto, jaco)= cv2.projectPoints(np.array([ p]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        cv2.circle(copiado, (int(ponto[0][0][0]), int(ponto[0][0][1])), 3, (0,255,0), -1)
        #cv2.putText(copiado,str(indexl[idx]),(int(ponto[0][0][0]), int(ponto[0][0][1])),cv2.FONT_HERSHEY_SIMPLEX,
        #                        0.6,(0,255,0),1)

    cv2.circle(copiado, center, 3, (255,0,0), -1)
    
    #p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    #nosep
    p1 = ( int(nosep[0][0][0]), int(nosep[0][0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    cv2.line(copiado, p1, p2, (255,0,0), 2)

# Display image
    cv2.imshow("Output", copiado)

def novoEquadR(rect,wib,heb,fatorenq=6):
    x1,y1,w,h = rect
    x2=x1+w
    y2=y1+h
    dx = (x2-x1)/fatorenq
    x1 = (x1 - dx) if (x1 - dx)>0 else 0
    x2 = (x2 + dx) if (x2 + dx)<wib else wib
    dy = (y2 -y1)/fatorenq
    y1 = (y1 - dy) if (y1 - dy)>0 else 0
    #y2 = (y2 + 2*dy) if (y2 + 2*dy) < heb else heb
    y2 = (y2 + dy) if (y2 + dy) < heb else heb
    return x1,y1,x2,y2

def detect(img, cascade):
    
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    #cv2.imshow("Cinza",gray)
    try:
        rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, # 3 2
                                     minSize=(120, 120))
    except:
        print "No process"
        return None
    
    h,w,c =img.shape
    
    for rect in rects:
        print "{}".format(rect)
        x1,y1,x2,y2 = novoEquadR(rect,w,h)
        print "{}; {}; {}; {}".format(x1,y1,x2,y2)
        newim = img[y1:y2,x1:x2].copy()
        bb = align.getLargestFaceBoundingBox(newim)
        if bb is None:
            continue
        hr,wr,c = newim.shape
        if wr > 600:
            newim = cv2.resize(newim,(0,0),fx=0.5,fy=0.5)
        return newim
    return None

if __name__ == '__main__':
    cascade = cv2.CascadeClassifier(join("..","..","data","haarcascades","haarcascade_frontalface_alt2.xml"))
    for pimagem in paths.list_images(pathimg):
        print pimagem
        im = cv2.imread(pimagem)
        if im is None:
            continue
        
        im=detect(im, cascade)
        if im is None:
            continue
        
        h,w,c = im.shape
        if h < 100 or w < 100:
            print "Imagem pequena??? -> {} -> {}x{}".format(pimagem,w,h)
            continue
        
        if im is None:
            continue
        
        analisaIm(im)
        while True:
            keyp = cv2.waitKey(0)& 0xFF
            if keyp == ord('q'):
                break
            elif keyp == ord('X'):
                model_points[5][0] -= 5
                model_points[6][0] += 5
                print "{}".format(model_points)
                analisaIm(im)
            elif keyp == ord('x'):
                model_points[5][0] += 5
                model_points[6][0] -= 5
                print "{}".format(model_points)
                analisaIm(im)
            elif keyp == ord('Y'):
                model_points[5][1] += 5
                model_points[6][1] += 5
                print "{}".format(model_points)
                analisaIm(im)
            elif keyp == ord('y'):
                model_points[5][1] -= 5
                model_points[6][1] -= 5
                print "{}".format(model_points)
                analisaIm(im)

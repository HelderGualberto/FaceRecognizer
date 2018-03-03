#!/usr/bin/python

# Autor: Roberto Kenji Hiramatsu
# Copyright: Universidade de Sao Paulo/Huawei
# Desempenho 
# Data: 2017/01/01 - Inicial
import cv2
import numpy as np
from os.path import join
from align_dlib import AlignDlib


align = AlignDlib(join("..","..","data","models","dlib","shape_predictor_68_face_landmarks.dat"))

# verifica a qualidade da imagem em escala de cinza com laplacian
def variance_of_laplacian(gray):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# extrai de imagem a regiao de interesse e a largura se detectar face
def extractFaceDLib(image):
    bb = align.getLargestFaceBoundingBox(image)
    #processa para separar somente rosto mesmo extraido de haar
    if bb is None:
        print "Face not found in dlib"
        return False,None,0
    else:
        samplec = image[bb.top():bb.bottom(),bb.left():bb.right()].copy()
        h,w,c = samplec.shape
        if h < 20 or w < 20 :
            print "Strange format in dlib"
            return False,None,0
        return True,samplec,w
    
    
# extrai de imagem a regiao de interesse e a largura se detectar face
def extractFaceDLibAli(image,size=96):
    bb = align.getLargestFaceBoundingBox(image)
    #processa para separar somente rosto mesmo extraido de haar
    if bb is None:
        print "Face not found in dlib"
        return False,None,0,None
    else:
        landmarks = align.findLandmarks(image, bb)
        aliImg = align.align(size, image, bb,
                    landmarks=landmarks,
                    landmarkIndices=align.OUTER_EYES_AND_NOSE)
        return True,aliImg,(bb.right()-bb.left()),landmarks
    
# gera mapa de magnetude do fft
def extractMagFFT(gray):
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    #magimg = np.log(np.abs(fshift)) #20*
    magimg=np.abs(fshift)
    return magimg

def fftScore(gray):
    magimg = extractMagFFT(gray)
    maxv = np.amax(magimg)
    #print "Max DC FFT:{}".format(maxv)
    thr = maxv/1000.0
    h,w = magimg.shape
    mthr = np.zeros(h*w)
    mthr = mthr.reshape((h,w))
    mthr.fill(thr)
    rcomp = magimg > mthr
    #rcomp = rcomp * 1.0
    score = np.sum(rcomp)/float(h*w)
    return score  

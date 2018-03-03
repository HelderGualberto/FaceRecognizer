#!/usr/bin/env python2
#
# Copyright 2015-2016 Escola Politecnica - Universidade de Sao Paulo
#
# Script para reformular svm


import os
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))
import argparse
import openface

from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC

from imutils import paths
#mypath='./people'
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import imagehash
from PIL import Image

import pickle

import RepUtil
from RepUtil import Face

modelDir = os.path.join(fileDir, '..', '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir,'treinado-jun16.t7'))
#'nn4.small2.v1.t7'))
#'treinado-jun16.t7'
#parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
#                    default=os.path.join(openfaceModelDir, 'custo9.3.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--cuda', type=bool, default=False)
parser.add_argument('--unknown', type=bool, default=False,
                    help='Try to predict unknown people')
parser.add_argument('--port', type=int, default=9000,
                    help='WebSocket Port')

parser.add_argument('--npath', type=str,
                    help="Caminho de processamento das imagens",
                    default='./people')


args = parser.parse_args()

mypath=args.npath

if args.unknown:
    unknownImgs = np.load("./examples/web/unknown.npy")
# os argumentos estao indicado para local existente a localizacao
print("args:",args)
align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                              cuda=args.cuda)

dpessoas = [f for f in listdir(mypath)]

#===============================================================================
# class Face:
#
#     def __init__(self, rep, identity,arquivo = None,triangs = None):
#         self.rep = rep
#         self.identity = identity
#         self.arquivo = arquivo
#         self.triangs = triangs
#
#     def __repr__(self):
#         return "{{id: {}, rep[0:5]: {}}}".format(
#             str(self.identity),
#             self.rep[0:5]
#         )
#===============================================================================

def getData(images):
    #para juntar as informacoes a serem treinadas com svm com valores formados por representacao
    X = []
    y = []
    for img in images.values():
        X.append(img.rep)
        y.append(img.identity)

    numIdentities = len(set(y + [-1])) - 1
    if numIdentities == 0:
        return None

    if args.unknown:
        numUnknown = y.count(-1)
        numIdentified = len(y) - numUnknown
        numUnknownAdd = (numIdentified / numIdentities) - numUnknown
        if numUnknownAdd > 0:
            print("+ Augmenting with {} unknown images.".format(numUnknownAdd))
            for rep in unknownImgs[:numUnknownAdd]:
                X.append(rep)
                y.append(-1)

    X = np.vstack(X)
    y = np.array(y)
    return (X, y)

def trainSVM(images):
    d = getData(images)
    if d is None:
        svm = None
    else:
        (X, y) = d
        numIdentities = len(set(y + [-1]))
        if numIdentities <= 1:
            return

        param_grid = [
                {'C': [1, 10, 100, 1000],
                 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000],
                 'gamma': [0.001, 0.0001],
                 'kernel': ['rbf']}
            ]
        svm = GridSearchCV(SVC(C=1), param_grid, cv=5).fit(X, y)
    return svm



if __name__ == '__main__':
    #print dpessoas
    identity = 0
    images = {}
    people = []
    for pessoa in dpessoas:
        print pessoa
        people.append(pessoa)
        totalim = 0
        imaceito = 0
        for pimagem in paths.list_images(mypath+'/'+pessoa):
            #print imagem
            #img = cv2.imread(pimagem)
            rgbFrame = cv2.imread(pimagem)
            totalim += 1
            #buf = np.fliplr(np.asarray(img))
            #rgbFrame = np.zeros((480, 640, 3), dtype=np.uint8)
            #rgbFrame[:, :, 0] = buf[:, :, 2]
            #rgbFrame[:, :, 1] = buf[:, :, 1]
            #rgbFrame[:, :, 2] = buf[:, :, 0]
            bb = align.getLargestFaceBoundingBox(rgbFrame)
            bbs = [bb] if bb is not None else []
            for bb in bbs:
                bl = (bb.left(), bb.bottom())
                tr = (bb.right(), bb.top())
                altura = bl[1]-tr[1]
                largura = tr[0]-bl[0]
                #if altura < 128 or largura <128:
                #    print 'rejeitado ',pimagem,\
                #        ' por largura ou altura menor que 128 a-> ',altura
                #    continue
                landmarks = align.findLandmarks(rgbFrame, bb)
                alignedFace = align.align(args.imgDim, rgbFrame, bb,
                                      landmarks=landmarks,
                                      landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                #if RepUtil.isBlur(alignedFace,125.0):
                #    print 'rejeitado ',pimagem,\
                #        ' por qualidade inferior a 125 valor b ',RepUtil.calcBlur(alignedFace)
                #    continue
                imaceito += 1
                rep = net.forward(alignedFace)
                vert = RepUtil.coletaVert(landmarks)
                phash = str(imagehash.phash(Image.fromarray(alignedFace)))
                images[phash] = Face(rep, identity,
                                     pimagem.split('/',3)[3],
                                     RepUtil.calcTriangAng(vert),
                                     RepUtil.definePosF(vert),
                                     RepUtil.calcBlur(alignedFace))
                #print images[phash]
        print imaceito,' do total de ',totalim,' para ',people[identity]
        identity += 1
    listaIm = images.values()
    nim= len(listaIm)
    ix = 0
    while ix < nim:
        jx = ix + 1
        while jx < nim:
            listaIm[ix].compara(listaIm[jx])
            jx += 1
        ix += 1

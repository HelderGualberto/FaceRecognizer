#!/usr/bin/python

# Autor: Roberto Kenji Hiramatsu
# Copyright: Universidade de Sao Paulo/Huawei
# Desempenho 
# Data: 2017/01/01 - Inicial
import numpy as np
import cv2
import math
from scipy.optimize import fsolve
from numpy.linalg import inv

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
    
#dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

def projecaoXYZ(rotationmatrix,translation_vector,camera_matrix,ponto):
    pr = np.matmul(rotationmatrix,ponto.T)
    pt = np.add(pr,translation_vector)
    pp = np.matmul(camera_matrix,pt)
    return pt,pp

# calculo usando processo de solvepnp do opencv
# @input  - sizeImg - a partir de atributo shape de imagem
# @output - angh,angv - angulo horizontal (azimute), angulo vertical ( altitude)
def solvePnPHPAng(landmarks,sizeImg,dist_coeffs = np.zeros((4,1))):
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
    
    center = (sizeImg[1]/2, sizeImg[0]/2)
    focal_length = max(sizeImg)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )    
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    rotationmatrix,jacobimat = cv2.Rodrigues(rotation_vector)
    #ponto1000 = np.array([(0.0, 0.0, 1000.0)])
    #ponto rotacionado
    #pt1,pp = projecaoXYZ(rotationmatrix,translation_vector,camera_matrix,ponto1000)
    #print "Projecao de {} do espaco de objeto para espaco de camera:\n {} ".format(ponto1000,pt1)    
    #print "No plano da imagem:\n {}".format(pp)
    #(nose_end_point2D, jacobian) = cv2.projectPoints(ponto1000, rotation_vector, translation_vector, camera_matrix, dist_coeffs)


    #ponto = np.array([(0.0, 0.0, 0.0)])
    #ponto rotacionado
    #pt2,pp = projecaoXYZ(rotationmatrix,translation_vector,camera_matrix,ponto)
    #print "Projecao de {} do espaco de objeto para espaco de camera:\n {} ".format(ponto,pt2)    
    #print "No plano da imagem {}".format(pp)
    pr0 = np.matmul(inv(rotationmatrix),-1.0*translation_vector)
    #vetorp = np.subtract(pt1,pt2)
    angh = -1.0*math.atan(pr0[0][0]/pr0[2][0])*180.0/math.pi
    angv = -1.0*math.atan(pr0[1][0]/pr0[2][0])*180.0/math.pi 
    return angh,angv

def calcDisDP(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    dx = float(x2-x1)
    dy = float (y2 - y1)
    return math.sqrt(dx*dx+dy*dy)

# funcao para obter o angulo posicionamento horizontal de um rosto
# regang valor de referencia (estimado) para angulo de triangulo formado entre as pontas da orelha e centro do rosto
# @return - retorna angulo em radianos
def calcHAngRosto(dre,drd,refang=math.pi/6.0):
    R = drd / dre
    #funcao trigonometrica para estabelecer relacao de projecao horizontal da imagem
    func  = lambda ang : R -((math.cos(refang)*math.cos(ang)-math.sin(refang)*math.sin(ang))/(math.cos(refang)*math.cos(ang)+math.sin(refang)*math.sin(ang)))
    ang_g = math.pi/4.0 if dre > drd else -math.pi/4.0
    ang_s = fsolve(func,ang_g)
    return ang_s

# calcular angulos da cabeca usando 3 pontos fiduciais + reserva da ponta do nariz modelo simplificado
# @input  - sizeImg - a partir de atributo shape de imagem
# @output - angh,angv - angulos em graus - angulo horizontal (azimute), angulo vertical ( altitude)

def calcHVAngRosto(landmarks,refang=math.pi/6.0):
    pre=landmarks[0]
    prd=landmarks[16]
    prc=landmarks[27]
    x1,y1 = pre
    x2,y2 = prd
    x3,y3 = prc
    #calcula angulo da reta p1 p2
    ah = (float(y2-y1)/float(x2-x1))
    #interseccao do eixo y
    bh = float(y1)- ah*float(x1)
    # para calcular as posicoes projetadas e distancia do centro as bordas
    if ah == 0.0:
        # considera que x1 e x2 estao no mesmo valor de y
        xp = x3
        yp = y1
        dre = float(x3 - x1)
        drd = float(x2 - x3)
        dpp = abs(y3 - yp)
    else:
        # calcula p3 projetado na reta p1 p2
        av = -1.0/ah
        bv = y3 - av * float(x3)
        xp = (bv - bh)/(ah - av)
        yp = ah * xp + bh
        dre = calcDisDP([xp,yp],pre)
        drd = calcDisDP([xp,yp],prd)
        dpp = calcDisDP([xp,yp],prc)
    # angulo de inclinacao horizonal
    angh = calcHAngRosto(dre,drd)
    # distancia normalizada horizontal
    ded = (dre+drd)/math.cos(angh)
    # distancia de plano de profundidade centro
    dpc = ded * math.sin(refang)/2.0
    # angulo de inclinacao vertical de rosto
    try:
        angv = math.asin(dpp/dpc)
    except ValueError:
        angv = -1.0
        print 'Erro de asin ',dpp,dpc
    angv = angv if yp > y3 else -angv
    return (angh[0]*180.0/math.pi),(angv*180.0/math.pi)
  

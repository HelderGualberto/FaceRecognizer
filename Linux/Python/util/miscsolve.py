import math
import numpy as np

#calcula distancia de dois pontos
def calcDistPoints(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    dx = float(x2-x1)
    dy = float (y2 - y1)
    return math.sqrt(dx*dx+dy*dy)

#calcula distancia euclidiana de forma clean
def calcDisR(a,b):
    subrep = np.subtract(a,b)
    vmrep = np.multiply(subrep,subrep)
    return vmrep.sum()

# funcao para enquadramento da area de busca
# rect - pontos P1 e P2 de retangulo
# w    - largura da imagem principal
# h    - altura da imagem principal
# fatorenq - fracao a ser extraida da imagem
def novoEquadR(rect,w,h,fatorenq=6):
    x1,y1,x2,y2 = rect
    dx = (x2-x1)/fatorenq
    x1 = (x1 - dx) if (x1 - dx)>0 else 0
    x2 = (x2 + dx) if (x2 + dx)<w else w
    dy = (y2 -y1)/fatorenq
    y1 = (y1 - dy) if (y1 - dy)>0 else 0
    y2 = (y2 + 2*dy) if (y2 + 2*dy) < h else h
    return x1,y1,x2,y2

def rectXY2XY(xywh):
    x,y,w,h = xywh
    return x,y,x+w,y+h

def rectXYHW(xy2xy):
    x1,y1,x2,y2 = xy2xy
    return x1,y1,x2-x1,y2-y1
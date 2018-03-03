import json
import argparse
import os
import cv2
import RepUtil
import numpy as np
import openface

parser = argparse.ArgumentParser()
parser.add_argument('--jsondb', type=str,
                    help="Armazenamento num arquivo json",
                    default='predicv2.json')

parser.add_argument('--npath', type=str,
                    help="Caminho de processamento das imagens",
                    default='/srv/openface/demos/web2/predic_proc')

fileDir = os.path.dirname(os.path.realpath(__file__))
parser.add_argument('--cascade', type=str,
                    help="cascade haar detector",
                    default=os.path.join(fileDir,'haarcascades','haarcascade_frontalface_alt.xml'))

modelDir = os.path.join(fileDir, '..', '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))


args = parser.parse_args()

caminho = args.npath

align = openface.AlignDlib(args.dlibFacePredictor)

def leDec(lido):
    #print lido
    lido = lido.replace("\r","").replace("\n","").replace("\0","")
    print lido
    try:
        decodif = json.loads(lido)
        print decodif
        return True,decodif
    except ValueError:
        print "invalido?",lido
        return False,None

def carregaLinhas():
    pessoas = {}
    with open(args.jsondb,'r') as fr:
        for lido in fr:
            ret,linha=leDec(lido)
            if ret:
                if pessoas.has_key(linha["name"]):
                    pessoas[linha["name"]].append(linha)
                else:
                    pessoas[linha["name"]]=[]
                    pessoas[linha["name"]].append(linha)
    return pessoas

def gravaLinhas(linhas):
    with open('rev'+args.jsondb,'a') as njsondb:
        for itens in linhas:
            RepUtil.gravaJSON(njsondb, itens)

def alteraEstado(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        lista,img = param
        ix = int (x/160)
        iy = int (y/180)
        conta = iy*8 + ix
        lt = (ix*160+2,iy*180+2)
        rb = ((ix+1)*160-2,(iy+1)*180-2)
        lista[conta]["isok"] = 0 if lista[conta]["isok"] == 1 else 1
        if lista[conta]["isok"] == 1:
            cv2.rectangle(img, lt , rb , color=(0,255,0),
                              thickness=3)
        else:
            cv2.rectangle(img, lt , rb , color=(0,0, 255),
                              thickness=3)
        print conta,lista[conta]["filename"]
        cv2.namedWindow(pessoa)
        cv2.setMouseCallback(pessoa,alteraEstado,param=(lista,img))
        cv2.imshow(pessoa,img)


def processaPessoa(pessoa,valores):
    print 'Processando ',pessoa
    largeimg = np.zeros((800,1280,3),np.uint8)
    xpos = 0
    ypos = 0
    #cv2.imshow(pessoa,largeimg)
    conta = 0
    subvalores = []
    for item in valores:
        if item["isok"] == 1:
            conta += 1
            subvalores.append(item)
            rgbIn = cv2.imread(caminho+'/'+item["filename"])
            [heb,wib,pb] = rgbIn.shape
            rects,bdisp=RepUtil.detectFace(rgbIn, cascade)
            for x1, y1, x2, y2 in rects:
                x1,y1,x2,y2 = RepUtil.novoEquad(x1,y1,x2,y2,wib,heb)
                vis_roi = rgbIn[y1:y2, x1:x2].copy()
                bb = align.getLargestFaceBoundingBox(vis_roi)
                bbs = [bb] if bb is not None else []
                if len( bbs) >0:
                    [he,wi,pro] = vis_roi.shape
                    if wi > 160:
                        fato = 160.0/float(wi)
                        rgbFrame  = cv2.resize(vis_roi,(0,0),fx=fato,fy=fato)
                    else:
                        rgbFrame = vis_roi.copy()
                    cv2.putText(rgbFrame,item["timestamp"],(5,150),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,0),1)
                    #cv2.putText(recorte,partes2[1],(5,15),cv2.FONT_HERSHEY_SIMPLEX,
                    #            0.8,(0,255,0),1)
                    [h,w,c] = rgbFrame.shape
                    #print h,w,c,largeimg.shape,xpos,ypos
                    largeimg[ypos:ypos+h,xpos:xpos+w]=rgbFrame
                    xpos += 160
                    if xpos > 1120:
                        xpos = 0
                        ypos += 180
                    if ypos >540:
                        cv2.namedWindow(pessoa)
                        cv2.setMouseCallback(pessoa,alteraEstado,param=(subvalores,largeimg))
                        cv2.imshow(pessoa,largeimg)
                        key = cv2.waitKey(0)
                        subvalores = []
                        largeimg = np.zeros((800,1280,3),np.uint8)
                        cv2.destroyWindow(pessoa)
                        conta = 0
                        ypos = 0
    if conta > 0:
        cv2.namedWindow(pessoa)
        cv2.setMouseCallback(pessoa,alteraEstado,param=(subvalores,largeimg))
        cv2.imshow(pessoa,largeimg)
        key = cv2.waitKey(0)
    cv2.destroyWindow(pessoa)

    conta = 0
    largeimg = np.zeros((800,1280,3),np.uint8)
    xpos = 0
    ypos = 0
    subvalores = []
    for item in valores:
        if item["isok"] == 0:
            subvalores.append(item)
            conta += 1
            rgbIn = cv2.imread(caminho+'/'+item["filename"])
            [heb,wib,pb] = rgbIn.shape
            rects,bdisp=RepUtil.detectFace(rgbIn, cascade)
            for x1, y1, x2, y2 in rects:
                x1,y1,x2,y2 = RepUtil.novoEquad(x1,y1,x2,y2,wib,heb)
                vis_roi = rgbIn[y1:y2, x1:x2].copy()
                bb = align.getLargestFaceBoundingBox(vis_roi)
                bbs = [bb] if bb is not None else []
                if len( bbs) >0:
                    [he,wi,pro] = vis_roi.shape
                    if wi > 160:
                        fato = 160.0/float(wi)
                        rgbFrame  = cv2.resize(vis_roi,(0,0),fx=fato,fy=fato)
                    else:
                        rgbFrame = vis_roi.copy()
                    cv2.putText(rgbFrame,item["timestamp"],(5,150),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,0),1)
                    cv2.putText(rgbFrame ,item["name"],(5,15),cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,(0,0,255),2)
                    [h,w,c] = rgbFrame.shape
                    #print h,w,c,largeimg.shape,xpos,ypos
                    largeimg[ypos:ypos+h,xpos:xpos+w]=rgbFrame
                    xpos += 160
                    if xpos > 1120:
                        xpos = 0
                        ypos += 180
                    if ypos >540:
                        cv2.namedWindow(pessoa)
                        cv2.setMouseCallback(pessoa,alteraEstado,param=(subvalores,largeimg))
                        cv2.imshow(pessoa,largeimg)
                        key = cv2.waitKey(0)
                        cv2.destroyWindow(pessoa)
                        subvalores = []
                        largeimg = np.zeros((800,1280,3),np.uint8)
                        conta = 0
                        ypos = 0
    if conta > 0:
        cv2.namedWindow(pessoa)
        cv2.setMouseCallback(pessoa,alteraEstado,param=(subvalores,largeimg))
        cv2.imshow(pessoa,largeimg)
        key = cv2.waitKey(0)
    cv2.destroyWindow(pessoa)

if __name__ == '__main__':
    cascade = cv2.CascadeClassifier(args.cascade)
    pessoas = carregaLinhas()
    for pessoa in pessoas.keys():
        processaPessoa(pessoa,pessoas[pessoa])
        gravaLinhas(pessoas[pessoa])

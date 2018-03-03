import cv2
import numpy as np
from miscsolve import calcDistPoints,novoEquadR
from align_dlib import AlignDlib
from os.path import join
import thread
from datetime import datetime

align = AlignDlib(join("..","..","data","models","dlib","shape_predictor_68_face_landmarks.dat"))

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.025,
                       minDistance = 5,
                       blockSize = 5 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#classe contendo informacoes da face rastreada formada por:
# ttl       - tempo de vida
# identity  - identidade corrente
# prevRect  - posicao da ultimo rectangulo
# prevFrame - contagem do frame corrente
# Adicionais center, e maxDist
class TrackedFace():
    def __init__(self,currentRect,currentFrame,identity,maxttl,grayimg=None,limMov=0.05):
        # tempo maximo de vida do rastreamento
        self.maxttl = maxttl
        # tempo de vida em numero de frames
        self.ttl = self.maxttl
        # identidade corrente atribuida na stream
        self.identity = identity
        self.prevRect = currentRect
        self.prevFrame = currentFrame
        # limite minimo de movimento para considerar rastreavel em termis de pixels
        self.limMov = limMov
        self.updateVAI(grayimg)
        # variavel para avisar se o objeto em questao esta vivo!!!
        self.goodOne = True
        #print "Criado objeto TrackedFace com identidade {} na posicao {}".format(self.identity,self.prevRect)
        #listagem de faces para uso em debug
        self.listHistory = []
        #Contagem de amostras
        self.contaSamp = 0

    # para atualizar variaveis adicionais e controle de rastreamento por Lucas-Kanade
    #fracmx o default era 4 teste com 2
    def updateVAI(self,grayimg,fracmx=2):
        x1,y1,x2,y2=self.prevRect
        # calculo de centro da face localizada
        self.center = ((x1+x2)/2,(y1+y2)/2)
        # distancia maxima esperada para localizar o mesmo rosto no tempo de vida
        self.maxDist =  (x2-x1)/fracmx
        self.initLK(grayimg)
        # variavel para controlar atualizacao por rastreamento
        self.tkLK = False



    def initLK(self,grayimg):
        #obtem forma da imagem na escala de cinza
        heb,wib = grayimg.shape
        #print "Trantando na area {}".format(self.prevRect)
        #gerar enquadramento para uso no rastreamento
        xn1,yn1,xn2,yn2 = self.prevRect
        w = xn2 - xn1
        dx = w / 6
        xn1 -= dx
        xn1 = xn1 if xn1 >0 else 0
        xn2 += dx
        xn2 = xn2 if xn2 < wib else wib
        #yn2 = (yn1+yn2)/2

        #= RepUtil.novoEquadR(self.prevRect, wib, heb)
        #print "Novo enquadramento inicial : ({},{}) ({},{}) para identidade {}".format(xn1,yn1,xn2,yn2,self.identity)
        #area de interesse para aplicar lucas kanade
        self.oldgray = grayimg[yn1:yn2,xn1:xn2].copy()
        #extrai os pontos para rastrear
        self.p0 = cv2.goodFeaturesToTrack(self.oldgray, mask = None, **feature_params)
        #self.drawGP()
        
    def drawGP(self):
        img = cv2.cvtColor(self.oldgray,cv2.COLOR_GRAY2BGR)
        
        if self.p0 is None:
            return
        
        corners = np.int0(self.p0)
        
        for i in corners:
            x,y = i.ravel()
            cv2.circle(img,(x,y),3,(0,0,255),-1)
        cv2.putText(img,'{}'.format(self.identity),(0,(20)),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,0,255),1)
        cv2.imshow("Good",img)
        cv2.waitKey(0)

    # compara se encontram na mesma posicao de detecao e remove o mais antigo para criar um novo grupo de identificacao
    def isSamePos(self,outro):
        x1,y1,x2,y2     = self.prevRect
        xo1,yo1,xo2,yo2 = outro.prevRect
        if x1 == xo1 and y1 == yo1 and x2 == xo2 and y2 == yo2:
            if self.identity < outro.identity:
                return self
            else:
                return outro

    # verifica se o retangulo do frame corrente refere-se a face e retorna verdadeiro se afirmativo
    def verify(self,rect,currentFrame):
        x1,y1,x2,y2 = rect
        newcenter = ((x1+x2)/2,(y1+y2)/2)
        dist = calcDistPoints(self.center,newcenter)
        if self.maxDist > dist:
            if self.prevFrame == currentFrame:
                print "A identidade {} na posiciao {} confrita com {}".format(self.identity,self.prevRect,rect)
                #atualiza a regiao para proposito do rastreamento
                self.prevRect=rect
                # gera contagem para eliminar duplo rastreamento
                self.ttl = 1
            else:
                self.prevRect=rect
                self.prevFrame=currentFrame
                # reinicia contagem do ciclo de vida da face
                self.ttl = self.maxttl
            return True
        return False

    # processa a existencia da face no frame corrente
    def trackLK(self,currentFrame,grayimg):
        #pode ser rastreavel dentro de 3 frames ap 30 fps!!
        if self.p0 is None:  # and (currentFrame-self.prevFrame) > 16
            # se pontos anteriores forem zero retorna sem procesar
            #print "Retornado pois p0 vazio???"
            return
        
        heb,wib = grayimg.shape
        #gerar enquadramento para uso no rastreamento
        x1,y1,x2,y2 = self.prevRect
        w = x2 - x1
        dx = w / 6
        xn1 = x1 - dx
        xn1 = xn1 if xn1 > 0 else 0
        xn2 = x2 + dx
        xn2 = xn2 if xn2 < wib else wib
        #self.drawGP()
        #yn1 = y1
        #yn2 = (y1+y2)/2
        #= RepUtil.novoEquadR(self.prevRect, wib, heb)
        #print "Novo enquadramento ({},{}) ({},{}) para identidade {}".format(xn1,yn1,xn2,yn2,self.identity)
        newgray= grayimg[y1:y2,xn1:xn2].copy()
        
                        
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.oldgray, newgray, self.p0, None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = self.p0[st==1]

        #pr1 = np.array((xn1,yn1))
        conta = 0
        soma  = 0
        somax = 0
        somay = 0
        #newgrayR = newgray.copy()
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            #print "Pontos (a,b) = ({},{}) (c,d) = ({},{})".format(a,b,c,d)
            #na,nb = np.array((a,b))+pr1
            #nc,nd = np.array((c,d))+pr1
            #na = int(a)
            #nb = int(b)
            #nc = int(nc)
            #nd = int(nd)
            #cv2.line(grayimg, (na,nb),(nc,nd), 0 , 2)
            #cv2.circle(newgrayR,(na,nb),5,255,-1)
            conta += 1
            soma  += calcDistPoints((a,b),(c,d))
            somax +=  a - c
            somay +=  b - d
        
        #=======================================================================
        # if newgray is not None:
        #     h,w = newgray.shape
        #     if w > 0 and h > 0:
        #         cv2.putText(newgrayR,'{}'.format(self.identity),(0,(20)),cv2.FONT_HERSHEY_SIMPLEX,
        #                         0.6,(255,255,255),1)
        #         cv2.imshow("New",newgray)
        #         cv2.imshow("old",self.oldgray)
        #         cv2.waitKey(0)
        #=======================================================================
        
        #para tratamento de regioes de nao face
        if conta > 1:
            media  = soma/conta
            #print "Tem media: {} em {} pontos e largura {} do individuo {}".format(media,conta,w,self.identity)
            if media < self.limMov: #
                # eliminando ponto por estar muto abaixo do esperado!!!
                self.ttl = 1
                self.goodOne = False
                #print "Retornado pois tem pouco movimento???"
                return
        #considerando que tenha mais de 4 pontos de controle
        if conta > 4:
            #media  = soma/conta
            mediax = int(somax/conta)
            mediay = int(somay/conta)
            #print "Individuo {} com pontos {}".format(self.identity,self.prevRect)
            #print "media:x {} y {} em {}".format(mediax,mediay,self.identity)
            x1 = x1+mediax
            x1 = x1 if x1 > 0 else 0
            y1 = y1+mediay
            y1 = y1 if y1 > 0 else 0
            hb,wb = grayimg.shape
            x2 = x2+mediax
            x2 = x2 if x2 < wb else wb
            y2 = y2+mediay
            y2 = y2 if y2 < hb else hb
            self.prevRect = (x1,y1,x2,y2)
            #print "Depois: {} em {}".format(self.prevRect,self.identity)
            #print "para pontos {}".format(self.prevRect)
            # seta para atualizar
            # considera para tracking se as posicoes x e y moverem
            #if mediax > 0 and mediay > 0:

            self.tkLK = True
            #disg   = math.sqrt((mediax*mediax)+(mediay*mediay))
            #ratiog = media/disg
            #print "Deslocou {} pp no individuo {} com {} ponto com vetor de deslocamento ({},{}) e razao de pp por g {}".format(media,self.identity,conta,mediax,mediay,ratiog)
        #else:
        #    print "Nao processado media pois tem poucos pontos"
            # Now update the previous frame and previous points
        #self.oldgray = newgray
        #self.p0 = good_new.reshape(-1,1,2)


    # faz update e se verdade mantem para o proximo ciclo
    def update(self,currentFrame,grayimg=None): #,fracmx=4
        #tratamento quando em modo de rastreio unificado  com ou sem atualizacao do tratador de track LK
        if self.prevFrame != currentFrame:
            self.ttl -= 1
            if self.ttl == 0:
                return False
            if self.tkLK:
                #nao diminui ttl pois apresenta possibilidade de rastreo
                #print "Atualizado por track objeto TrackedFace com identidade {} na posicao {} com ttl {}".format(self.identity,self.prevRect,self.ttl)
                self.updateVAI(grayimg) #,fracmx
                return True

        # se nao for bom entao deve ser eliminado pois tem pouco movimento
        if not self.goodOne:
            return False
        #if self.ttl == 0:
        #    return False
        #else:
            #x1,y1,x2,y2 = self.prevRect
            #self.center = ((x1+x2)/2,(y1+y2)/2)
            #self.maxDist =  (x2-x1)/fracmx
        self.updateVAI(grayimg) #,fracmx

        return True

    # coleta a amostra da imagem de ratreamento
    # img          - imagem colorida em tamanho original
    # currentframe - contador do frame corrente
    # fator        - reducao aplicada para detecao da face
    # maxWidth     - largura maxima da imagem coletada
    # minWidth     - largura minima da face
    # faceAli      - processa alinhamento da face
    # retorna (isCap,seq,cropimg) -
    #          isCap   - indicador que a imagem foi coletada
    #          seq   - contagem do coletor
    #          cropimg - imagem capturada
    def collectSample(self,img,currentframe,faceAli=False,fator=1.0,maxWidth=180.0,minWidth=107.0): #era maxWidth=120.0
        # Obtem tamaho original na imagem
        x1,y1,x2,y2 = (np.array(self.prevRect)/fator).astype(int)
        # obtem o valor da largura da imagem
        ws = abs(x2-x1)
        # retorna se for menor que minWidth. Isto e, a imagem esta abaixo no nivel para rede com 96
        #print "ws:{}".format(ws)
        if ws < minWidth:
            return False,self.contaSamp,None
        #fator de redimencionamento
        fatorS = 1.0  if ws < maxWidth else maxWidth/float(ws)

        h,w,c = img.shape
        # obtendo os valores da janela de recorte a ser usada no dlib
        x1,y1,x2,y2 = novoEquadR((x1,y1,x2,y2),w,h)
        #recortando a regiao da imagem
        sample = img[y1:y2,x1:x2].copy()
        
        
        # redimencionado para faixa adequada para processamento na dlib
        sample = cv2.resize(sample,(0,0),fx=fatorS,fy=fatorS)

        #processamento adicional para desentlelacar
        #sample,ret = separaOddEven(sample)
        #if ret:
        #    cv2.imshow("Ajusta sample",sample)
        #else:
        #if not ret:
        #    return False,self.contaSamp,None
        # processa dlib
        bb = align.getLargestFaceBoundingBox(sample)
        #processa para separar somente rosto mesmo extraido de haar
        if bb is None:
            return False,self.contaSamp,None

        wb = abs(bb.right()-bb.left())
        hsr,wsr,c = sample.shape
        if float(wb)/float(wsr) < 0.6:
            print "Recover image sample because face in sample image is small"

            xe1 = x1+int(bb.left()/fatorS)
            ye1 = y1+int(bb.top()/fatorS)
            xe2 = x1+int(bb.right()/fatorS)
            ye2 = y1+int(bb.bottom()/fatorS)
            x1,y1,x2,y2 = novoEquadR((xe1,ye1,xe2,ye2),w,h)
            #recortando a regiao da imagem
            sample = img[y1:y2,x1:x2].copy()
            fatorS = 1.0
            wb= abs(x2 - x1)
            if faceAli:
                bb = align.getLargestFaceBoundingBox(sample)

        if faceAli:
            if bb is not None:
                rwi = (wb/(ws*fatorS))
                if  rwi < 0.7 or rwi >1.4:
                    return False,self.contaSamp,None
                landmarks = align.findLandmarks(sample, bb)
                sample = align.align(96, sample, bb,
                                      landmarks=landmarks,
                                      landmarkIndices=align.OUTER_EYES_AND_NOSE)
            else:
                #se nao encontrar face entao deve ser desconsiderado
                return False,self.contaSamp,None

        #arq = "samp_{:04d}_{:02d}_{:06d}.jpg".format(self.identity,self.contaSamp,currentframe)
        #cv2.imwrite(arq,sample)
        #listagem de faces para uso em debug
        self.listHistory.append((currentframe,sample))
        #cv2.putText(sample,str(self.identity),(0,20),cv2.FONT_HERSHEY_SIMPLEX,
                                #0.6,(0,255,255),1)
        #cv2.imshow("Sample",sample)
        #cv2.waitKey(0)
        #Contagem de amostras
        self.contaSamp += 1
        # retorna a contagem atual de amostras
        return  True,self.contaSamp,sample

# classe para controlar tracking das faces
# maxttl - maximo numero de frames para tempo de rastreamento em 30fps o valor 10 represetenta 330ms
# maxttl - maximo numero de frames para tempo de rastreamento em 30fps o valor 15 represetenta 500ms
class TrackingFaces():
    def __init__(self,countIdentity = 0,maxttl=15):
        #lista de faces rastreadas
        self.listTrackedFaces = []
        #contador de identidade
        self.countIdentity = 0
        # tempo maximo de rastreamento em frames
        self.maxttl = maxttl
        # contador para verificar boas amostragens
        self.countGood = 0

    #processamento do ratreamento atraves de lista de retangulos no formato posicao incial largura altura
    def updateFromRectWH(self,currentFrame,rects,gray):
        for tf in self.listTrackedFaces:
            tf.trackLK(currentFrame,gray)

        newlistTrackedFaces = []
        listToRemoveDup = []
        listLabedFaces = {}
        for x1,y1,w,h in rects:
            # verificador para nova identidade se nao encontrar uma identidade
            isNotFound = True
            # caso encontre anterior
            prevTf = None
            listLabedFaces[(x1,y1,w,h)]=[]
            for tf in self.listTrackedFaces:
                if tf.verify((x1,y1,x1+w,y1+h),currentFrame):
                    # atualiza os pontos de rastreamento na face localizada
                    tf.initLK(gray)
                    if not isNotFound:
                        print "Anterior {} na posicao {} confrita com {} na posicao {}".format(prevTf.identity,prevTf.prevRect,tf.identity,tf.prevRect)
                        listToRemoveDup.append(prevTf) #tf.isSamePos(prevTf))
                        # remove tambem a atual da lista
                        listToRemoveDup.append(tf)
                    # processa caso encontre uma identidade na lista
                    isNotFound = False
                    prevTf = tf
                    listLabedFaces[(x1,y1,w,h)].append(tf.identity)
            if isNotFound:
                # processa caso nao encontre correspondente anterior
                newlistTrackedFaces.append(TrackedFace((x1,y1,x1+w,y1+h),currentFrame,self.countIdentity,self.maxttl,grayimg=gray))
                listLabedFaces[(x1,y1,w,h)].append(self.countIdentity)
                self.countIdentity += 1
        for tf in self.listTrackedFaces:
            # para atualizar ciclo de vida das faces rastreadas
            if tf.update(currentFrame,grayimg=gray) and tf not in listToRemoveDup:
                #processa se ciclo de vida da face rastreada nao espirou
                newlistTrackedFaces.append(tf)
            else:
                if tf.contaSamp > 3:
                    self.countGood += 1
                    print "Contado {} faces com mais de tres amostras".format(self.countGood)
                #print "Individuo {} foi removido da lista de rastreamento".format(tf.identity)
        # atualiza a lista de faces rastreadas
        self.listTrackedFaces = newlistTrackedFaces
        return listLabedFaces

    #obter a lista de faces em track state
    def getOnTrackFaces(self,currentFrame):
        listRects = []
        listaNomes = {}
        for tf in self.listTrackedFaces:
            if tf.prevFrame != currentFrame:
                x1,y1,x2,y2 = tf.prevRect
                w = x2 - x1
                h = y2 - y1
                listRects.append((x1,y1,w,h))
                listaNomes[(x1,y1,w,h)]=[]
                listaNomes[(x1,y1,w,h)].append((tf.identity,tf.ttl))

        return listRects,listaNomes

    #obter historico de imagens de face em rastreo
    def drawSampleTrack(self):

        #numero de linhas de amostras
        n = len(self.listTrackedFaces)
        if n < 1:
            print "Retorna pois nao tem imagens armazenadas"
            return

        # imagem
        largeimg = np.zeros((120*n+120,1200+120,3),np.uint8)
        ypos = 0

        for tf in self.listTrackedFaces:
            if tf.contaSamp < 2:
                continue
            xpos = 0
            nim = len(tf.listHistory)
            print "nim: {}".format(nim)
            idxc = 0
            for idx,(cframe,img) in enumerate(tf.listHistory):
                h,w,c = img.shape
                #print "Origem: {}".format(img.shape)
                if (nim-idxc) < 10:
                    largeimg[ypos:ypos+h,xpos:xpos+w]=img
                    cv2.putText(largeimg,str(cframe),(xpos,ypos+60),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),1)
                    xpos += 120
                else:
                    tf.listHistory.remove((cframe,img))
                idxc += 1
                #print "idxc:{}".format(idxc)

            cv2.putText(largeimg,str(tf.identity),(0,ypos+20),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,255),3)

            ypos += 120
        if ypos > 0:
            cv2.imshow("Sample",largeimg)
            #cv2.waitKey(0)
        #cv2.waitKey(0)

    #coletor de imagens
    # currentFrame - contagem do frame atual
    # fator        - reducao de escala usada para melhorar o desempenho de localizacao de face na imagem
    #                e referencia para extracao da face em escala na imagem original.
    def collectCurrentSamp(self,currentFrame,img,fator,mdb=None,base="camara",faceAlinhada = False):
        retop = False
        for tf in self.listTrackedFaces:
            isCap,seq,cropimg = tf.collectSample(img,currentFrame,faceAli=faceAlinhada,fator=fator)
            if mdb is not None:
                if seq > 1: # a partir de duas faces
                    # somente processar se for a terceira imagem da sequencia
                    if seq == 2 and isCap:
                        #armazenar a imagem anteriores da sequencia
                        #gravaIm(self,img,base,seq,frame,datahora,depessoa,alinhado)
                        for idx,(cframe,sample) in enumerate(tf.listHistory):
                            thread.start_new_thread(mdb.gravaIm,(sample, base, idx, cframe, datetime.now(), tf.identity, faceAlinhada))
                            retop = True
                    else :
                        if isCap:
                            thread.start_new_thread(mdb.gravaIm,(cropimg, base, seq, currentFrame, datetime.now(), tf.identity, faceAlinhada))
                            retop = True
        return retop

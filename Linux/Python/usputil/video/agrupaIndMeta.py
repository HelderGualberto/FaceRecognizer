#!/usr/bin/python
# Autor: Roberto Kenji Hiramatsu
# Tratador para agrupar dados gerados pelo testFile.py e anotaTUV.py. Usando o metadado gerado e o principio
# de agrupamento em testeFile.py
# Data inicial: 2016-09-04
#
import json
import argparse
import os

import externoexec4

parser = argparse.ArgumentParser()
parser.add_argument('--jsondb', type=str,
                    help="Armazenamento num arquivo json",
                    default='d:\\temp\\teste4\\meta.json')
args = parser.parse_args()

def leDec(lido):
    #print lido
    lido = lido.replace("\r","").replace("\n","").replace("\0","")
    #print lido
    try:
        decodif = json.loads(lido)
        #print decodif
        srep = decodif["netrep"].replace("[","").replace("]","").split(",")
        rep = {}
        for idx,v in enumerate(srep):
            valor = float(v)
            if valor != 0:
                rep[idx]=valor
            #else:
            #    print idx,valor
        decodif["netrep"]=rep
        #print decodif["netrep"]
        return True,decodif
    except ValueError:
        #print "invalido?",lido
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

def calcDistancia(r1,r2):
    soma = 0.0
    for idx, val in enumerate(r1):
        pa=float(r1[idx])-float(r2[idx])
        soma += pa * pa
    return soma
#tdis=0.09,tang=0.15):
def pertence(pessoa1,pessoa2,tdis=0.09,tang=0.15):
    tangse = 1.5*tang
    tdisse = tdis/2.0
    distref = 4.0
    anghref = 2.0
    angvref = 2.0
    for idx,valor in enumerate(pessoa1):
        for idx2,valor2 in enumerate(pessoa2):
            distancia = calcDistancia(valor["netrep"],valor2["netrep"])
            dangh = abs(valor["anghori"]-valor2["anghori"])
            dangv = abs(valor["angvert"]-valor2["angvert"])
            if distref > distancia and distancia != 0.0 and anghref > dangh and angvref > dangv:
                distref = distancia
                anghref = dangh
                angvref = dangv
            if (dangh < tang and dangv <tang and distancia <tdis) or (dangh < tangse and dangv < tangse and distancia < tdisse ):
                #print idx,":",idx2," ",valor["filename"],valor2["filename"]," --> d= ",distancia," ah,av ",dangh,dangv
                #print "Minimo d,h,v ",distref,anghref,angvref
                return True
    return False
if __name__ == '__main__':
    with open(args.jsondb,'r') as fr:
        pessoas = carregaLinhas()
        ordenado = sorted(pessoas,key=lambda pessoa:-len(pessoas[pessoa]))
        #Cria lista de tratados
        tratado = {}
        for idx,p in enumerate(ordenado):
            tratado[idx] = False
        for idx1,p1 in enumerate(ordenado):
            if tratado[idx1]:
                continue
            tratado[idx1]=True
            agrupado = []
            for idx2,p2 in enumerate(ordenado):
                if idx2 < idx1 or tratado[idx2]:
                    continue
                if pertence(pessoas[p1],pessoas[p2]):
                    tratado[idx2] = True
                    agrupado.append(p2)
                    #print p1," pode ser o mesmo de ",p2
            if len(agrupado) > 0:
                destino = "D:\\temp\\teste4\\i{}".format(p1)
                pOri="/home/yakumo/openface/huawei/usputil/video4"
                if externoexec4.pastaOk(destino):
                    print "Copiando individuo {} em {}".format(p1,destino)
                    externoexec4.copia(p1, destino,pastaOri=pOri)
                print p1," pode ser o mesmo de ",agrupado
                for outro in agrupado:
                    if externoexec4.pastaOk(destino):
                        print "Copiando individuo {} em {}".format(outro,destino)
                        externoexec4.copia(outro, destino,pastaOri=pOri)

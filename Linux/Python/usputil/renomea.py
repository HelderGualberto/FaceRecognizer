#!/usr/bin/python

# Autor: Roberto Kenji Hiramatsu
# Copyright: Universidade de Sao Paulo/Huawei
# Script desenvolvido para uniformizar nome dos arquivos com base no diretorio raiz para imagens do tipo .jpg
# Data: 2016/08/17


import os,sys
import re

nargs = len(sys.argv)
print 'Num args ',nargs
print 'Args ',sys.argv

if(nargs<2):
    mypath = '../localhighn'
else:
    mypath = sys.argv[1]
print 'Processando diretorio ',mypath

def renomear(pathb,nameb):
    arquivos = os.listdir(pathb+'/'+nameb)
    conta = 1
    expressao = '{}_\d+.jpg'.format(nameb)
    parte = re.compile(expressao)
    for arq in arquivos:
        print 'Processando',arq
        if parte.search(arq):
            print arq,' esta no padrao'
        else:
            novonom = "{}/{}/{}_{:03d}.jpg".format(pathb,nameb,nameb,conta)
            antinom = "{}/{}/{}".format(pathb,nameb,arq)
            while os.path.isfile(novonom):
                print 'Ja existe ',novonom
                conta += 1
                novonom = "{}/{}/{}_{:03d}.jpg".format(pathb,nameb,nameb,conta)
            print 'Renomea de ',antinom,'para',novonom
            os.rename(antinom,novonom)
            conta += 1

if __name__=="__main__":
    dirs = os.listdir(mypath)
    for dirn in dirs:
        print dirn
        renomear(mypath,dirn)

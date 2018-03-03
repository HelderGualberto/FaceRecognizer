from imutils import paths
import argparse
from os import listdir

mypath='/srv/data/database_de_imagens_02-08-2016'

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help="Caminho de levantamento de estatistica",
                    default='./people')

args = parser.parse_args()
print args.path

mypath = args.path

dpessoas = [f for f in listdir(mypath)]

histo = {}
contav=0
for nome in dpessoas:
    #print nome
    listaa = [sf for sf in listdir(mypath+'/'+nome)]
    tam = len(listaa)
    if histo.has_key(tam):
        histo[tam] += 1
    else:
        histo[tam] = 1
    if tam >2:
        contav += 1
    #else:
    #    print nome,' com tam ',tam


for k in histo.keys():
    print k,':',histo[k]
print contav, ' de total de ',len(dpessoas)

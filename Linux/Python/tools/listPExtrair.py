from pymongo import MongoClient
client = MongoClient("mongodb://192.168.10.236:37027")
db = client["corredor"]

cols = db.collection_names()

listaF = []
listaC = []
for nc in cols:
   parts =nc.split('_')
   #print parts[1]
   # verifica se tem a parte frame ou cand
   if len(parts)>2:
      if parts[2] == 'frame':
         listaF.append(parts[1])
         #print "{} e tipo sequencia de frame".format(nc)
      elif parts[2] == 'cand':
         listaC.append(parts[1])
         #print "{} e tipo candidatos a face".format(nc)
   #else:
      #print "{} e localizacao de face na referencia dlib".format(nc)
listaP = []
texto  = " "
for base in listaF:
    if base not in listaC:
        #print "{} nao tem processamento de face".format(base)
        listaP.append(base)
        texto = "{} {} ".format(texto,base)
print texto

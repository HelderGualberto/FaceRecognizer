from pymongo import MongoClient
client = MongoClient("mongodb://192.168.10.236:37027")
db = client["corredor"]

cols = db.collection_names()

for nc in cols:
   parts =nc.split('_')
   print parts[1]
   # verifica se tem a parte frame ou camd
   if len(parts)>2:
      if parts[2] == 'frame':
         print "{} e tipo sequencia de frame".format(nc)
      elif parts[2] == 'camd':
         print "{} e tipo candidatos a face".format(nc)
   else:
      print "{} e localizacao de face na referencia dlib".format(nc)

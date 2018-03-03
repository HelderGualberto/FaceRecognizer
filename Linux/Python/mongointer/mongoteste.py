from pymongo import MongoClient

#client = MongoClient("mongodb://nuvemusp.pad.lsi.usp.br:37027")
client = MongoClient("mongodb://videobroker.pad.lsi.usp.br:37027")
#db = client.test
db = client.gridfs

#retcur = db.face.find()
retcur = db.fs.files.find()

for document in retcur:
  print(document)




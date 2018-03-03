from pymongo import MongoClient
import gridfs

client = MongoClient("mongodb://nuvemusp.pad.lsi.usp.br:37027")
db = client.gridfs

retcur = db.fs.files.find({'mesdia':908})
#,'horaminuto':{$gte:949}

fs = gridfs.GridFS(db)
#fs = GridFSBucket(db)
for document in retcur:
  print document['mesdia'],document['horaminuto'],document['filename']
  if document['horaminuto'] >= 600:
      arquivo = open(document['filename'],"wb")
      arquivo.write(fs.get(document['_id']).read())
      print "Gravado ",document['filename']
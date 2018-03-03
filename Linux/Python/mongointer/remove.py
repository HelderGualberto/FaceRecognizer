from pymongo import MongoClient
import gridfs

client = MongoClient("mongodb://videobroker.pad.lsi.usp.br:37027")
#client = MongoClient("mongodb://nuvemusp.pad.lsi.usp.br:37027")
db = client.gridfs
#{'mesdia':908}
retcur = db.fs.files.delete_many({})
print "Removidos ",retcur.deleted_count
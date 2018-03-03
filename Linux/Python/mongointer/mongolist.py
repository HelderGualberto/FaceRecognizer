from pymongo import MongoClient
import gridfs

#client = MongoClient("mongodb://nuvemusp.pad.lsi.usp.br:37027")
#client = MongoClient("mongodb://videobroker.pad.lsi.usp.br:37027")
# acesso a base de detecao de face por via interna
client = MongoClient("mongodb://10.20.0.253:27017")

db = client.gridfs

#{'mesdia':908}
retcur = db.fs.files.find({'mesdia':1010})


fs = gridfs.GridFS(db)
#fs = GridFSBucket(db)
conta = 0
soma = 0
for document in retcur:
    if "mesdia" in document.keys():
        print "{:04d}; {:04d}; {:9d}; {}; {}; {}".format(document["mesdia"],document["horaminuto"],document["frame"],
                                                 document['filename'],document['area'],document['framebase'])
        #print document.keys()
        soma += document["length"]
        conta += 1
print "Total de ",conta," imagens e ",(soma/1048576)


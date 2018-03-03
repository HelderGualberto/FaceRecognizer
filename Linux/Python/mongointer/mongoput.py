from pymongo import MongoClient
import gridfs

#client = MongoClient("mongodb://nuvemusp.pad.lsi.usp.br:37027")
client = MongoClient("mongodb://videobroker.pad.lsi.usp.br:37027")
db = client.gridfs
#db.create_collection("teste")
fs = gridfs.GridFS(db)
fileId = fs.put(open(r'D:/projetos/Safety_City/Code/Python/capturarstp/face_09.08.20.05.42_f000000280_1.jpg','rb'),filename="frame_09.06.19.00.46_f00011692.jpg",localidade="aqui",quando="hoje",area=(1,2,3,4))
out = fs.get(fileId)
print out.length

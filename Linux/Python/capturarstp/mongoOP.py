from pymongo import MongoClient
import gridfs

client = MongoClient("mongodb://nuvemusp.pad.lsi.usp.br:37027")
db = client.gridfs
#db.create_collection("teste")
fs = gridfs.GridFS(db)
fileId = fs.put(open(r'D:/projetos/Safety_City/Code/Python/capturarstp/frame_09.06.19.00.46_f00011692.jpg','rb'),filename="frame_09.06.19.00.46_f00011692.jpg",localidade="aqui",quando="hoje")
out = fs.get(fileId)
print out.length

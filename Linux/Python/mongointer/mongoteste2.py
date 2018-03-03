from pymongo import MongoClient

#client = MongoClient("mongodb://nuvemusp.pad.lsi.usp.br:37027")
client = MongoClient("mongodb://videobroker.pad.lsi.usp.br:37027")
#db = client.test
db = client.teste

#retcur = db.face.find()
#retcur = db.item.insert({"chave1":"alguma coisa","chave2":"outra coisa"})

retcur = db.item.update({"chave1":"alguma coisa"},{"$set":{"chave3":"mais uma coisa coisa"}})

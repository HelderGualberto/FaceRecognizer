from pymongo import MongoClient


client = MongoClient("mongodb://192.168.174.129:27017")
db = client["teste"]

colecao=db["operacao"]

# for idx in range(10):
#     info = {}
#     info['chave']=1234
#     info['idx']=idx
#     colecao.insert_one(info)


info = {}
info['chave']=1234
colecao.update(info, {"$set": {'comment':'processamento em multiplos documentos'}}, upsert=False,multi=True)

ret = colecao.find()

for i in ret:
    print i
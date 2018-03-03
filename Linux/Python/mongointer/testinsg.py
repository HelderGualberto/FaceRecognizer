from pymongo import MongoClient


client = MongoClient("mongodb://192.168.174.129:27017")
db = client["teste"]

colecao=db["operacao"]

info = {}
info['chave']=1234
# info['texto']='primeira entrada'
# colecao.insert_one(info)
# print info
# info={}
# info['chave']=2345
# info['texto']='segunda entrada'
# colecao.insert_one(info)
# print info
post = colecao.find_one(info)

print post

colecao.update({'_id':post['_id']}, {"$set": {'comment':'um comentario para'}}, upsert=False)
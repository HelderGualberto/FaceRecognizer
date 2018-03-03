from pymongo import MongoClient

client = MongoClient("mongodb://192.168.10.236:37027")
db = client["corredor"]
colecao = db["teste"]


colecao.insert_one({"fileIn":"file1234.jpg"})

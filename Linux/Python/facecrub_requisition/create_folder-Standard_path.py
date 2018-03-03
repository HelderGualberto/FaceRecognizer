import os

def build_std_path(name,_id):
    size = len(_id)
    path = _id
    
    while size < 5:
        path = "0" + path
        size += 1
    return path

name = "Ator qualquer"

path = "c:/" + name

_id = "50"

path = build_std_path(name,_id)
print path

try:
    os.makedirs(path)
except: 
    print "Error while creating folder"
    Exception


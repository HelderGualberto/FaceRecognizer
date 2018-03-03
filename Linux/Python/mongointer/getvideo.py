from pymongo import MongoClient
import gridfs
import cv2
import os

client = MongoClient("mongodb://192.168.10.236:27017")
db = client['kurento']

retcur = db.fs.files.find()
#,'horaminuto':{$gte:949}

fs = gridfs.GridFS(db)
#fs = GridFSBucket(db)
conta = 0
for document in retcur:
    arquivo = "D:\\projetos\\Safety_City_offdata\\teste\\v{:03d}{}.mp4".format(conta,document['filename'])
    conta += 1
    parq = open(arquivo,"wb")
    parq.write(fs.get(document['_id']).read())
    parq.close()
    print "Gravado {}".format(arquivo)
    
    cap = cv2.VideoCapture(arquivo)
    temframe=True
    contaf = 0
    while(temframe):
        temframe, frame = cap.read()
        contaf += 1
        try:
            reduzido = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
            cv2.imshow(arquivo,reduzido)
            key = cv2.waitKey(10)
        except:
            print "Erro ???"
        if key & 0xFF == ord('q'):
            cap.release()            
            break
        elif key & 0xFF == ord('d'):
            cap.release()
            os.remove(arquivo)
            break
    if contaf < 18 and key & 0xFF != ord('q'):
        cap.release()
        os.remove(arquivo)
    cv2.destroyWindow(arquivo)
    
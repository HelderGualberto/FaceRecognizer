import threading
import time

class myThread(threading.Thread):

    def __init__(self,threadID,name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name

    def run(self):
        func()

def func ():
    count = 0
    while True:
        print count
        time.sleep(1)
        if count == 100:
            break
        count +=1
    

#main function

_thread = myThread(1,"thread")
_thread.run()
_thread.join(5)


while True:
    pass
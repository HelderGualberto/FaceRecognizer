import multiprocessing
import time
import thread

class my_process(multiprocessing.Process):

    def kill_process(self,delay):
        time.sleep(delay)
        self.terminate()

    def time_out(self,delay):
        thread.start_new_thread(self.kill_process,(delay,))

def func (value):
    count = 0
    while True:
        
        time.sleep(1)
        print value
        if count == 15:
            break
        count +=1
 

#main function
if __name__ == '__main__':
    delay = 5

    while True:

        while len(multiprocessing.active_children()) > 4:
            pass

        print len(multiprocessing.active_children())

        p = my_process(target=func,args=(delay,))
        p.start()
        
        p.time_out(delay)
        delay += 0.1


    

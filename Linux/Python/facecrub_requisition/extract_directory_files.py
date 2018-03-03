import os, re, subprocess, multiprocessing

MAX_OF_THREADS = 10

class my_process(multiprocessing.Process):

    def kill_process(self,delay):
        time.sleep(delay)
        self.terminate()

    def time_out(self,delay):
        thread.start_new_thread(self.kill_process,(delay,))

def extract_files(file):
	
	#Create a reference to the origin file path + file name
	file_origin = root_path+"/"+file
	print "File origin: " + file_origin

	try:
		os.makedirs(output_path)
	except Exception:
		pass

	#Create a reference to the destiny file path
	file_destiny = output_path+"/"+file
	print "File detiny: " + file_destiny
	#Call a linux process to copy the file from origin to destiny path
	
	try:
		subprocess.call(["mv",file_origin,file_destiny])
	except Exception:
		pass
	#Create a reference to the file to extract path
	file_name_path = output_path +"/"+ file
	print "name: " + file_name_path
	#extract the file
	p = subprocess.Popen(["bzip2","-d",file_name_path],stdout=subprocess.PIPE)
	(output,err) = p.communicate()
	#file_log.write(output)



def main():
	for i in range(0,len(files)):
		
		if __name__ == '__main__':
			while len(multiprocessing.active_children()) > MAX_OF_THREADS:
				pass

			p = my_process(target=extract_files,args=(files[i],))
			p.start()

		else:
			print "Process ERROR!"

		#try:


root_path = "C:/Users/helder.rodrigues/Downloads/pre"
#input("Enter the root path: ")
output_path = "C:/Users/helder.rodrigues/Downloads/pre1"
#input("Enter the output path: ")


files = os.listdir(root_path)
#file_log = open("./log_files/extract_log.txt","w")

print root_path
print output_path


main()






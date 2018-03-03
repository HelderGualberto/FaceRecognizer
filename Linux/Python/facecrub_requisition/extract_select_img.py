import os, re, subprocess, multiprocessing

MAX_OF_THREADS = 10

RE = "[0-9]{5}_[0-9]{6}_(f|h|r|q)[a-z](.){5}bz2"

root_path = "./colorferet/dvd2/data/images/"

root_folders = os.listdir(root_path)

output_path = "./Images/colorferet/cd2"

lexical_analiser = re.compile(RE)

file = open("./extract_log.txt","w")

class my_process(multiprocessing.Process):

    def kill_process(self,delay):
        time.sleep(delay)
        self.terminate()

    def time_out(self,delay):
        thread.start_new_thread(self.kill_process,(delay,))

def extract_files(i,j,sub_path,folder_files):
	if lexical_analiser.search(folder_files[j]):
		folder_name = output_path +"/"+ root_folders[i]
		print folder_name
		try:				
			os.makedirs(folder_name)
		except:
			pass
		#Create a reference to the origin file path + file name
		file_origin = sub_path+"/"+folder_files[j]
		print "File origin: " + file_origin
		#Create a reference to the destiny file path
		file_destiny = output_path+"/"+root_folders[i]
		print "File detiny: " + file_destiny
		#Call a linux process to copy the file from origin to destiny path
		
		try:
			subprocess.call(["mv",file_origin,file_destiny])
		except Exception:
			pass
		#Create a reference to the file to extract path
		file_name_path = file_destiny +"/"+ folder_files[j]
		print "name: " + file_name_path
		#extract the file
		p = subprocess.Popen(["bzip2","-d",file_name_path],stdout=subprocess.PIPE)
		(output,err) = p.communicate()
		file.write(output)



def main():
	print len(root_folders)
	for i in range(0,len(root_folders)):
		sub_path = root_path + root_folders[i]
		folder_files = os.listdir(sub_path)
		for j in range(0,len(folder_files)):
			
			if __name__ == '__main__':
				while len(multiprocessing.active_children()) > MAX_OF_THREADS:
					pass

				p = my_process(target=extract_files,args=(i,j,sub_path,folder_files,))
				p.start()

			else:
				print "Process ERROR!"

			#try:
		
main()





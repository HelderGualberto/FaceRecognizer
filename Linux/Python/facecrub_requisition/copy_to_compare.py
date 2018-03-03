import os, subprocess,sys,csv,multiprocessing


#the first index os argv is the name of the scrip.py
root_path = sys.argv[1] #insert root path
out_path = sys.argv[2] #insert output file

print root_path
print out_path

csv_file = open("../src/compare_img.csv")
files = csv.reader(csv_file)


MAX_OF_THREADS = 9

def compare(file):
	img1 = root_path + "/" + file[0]
	img2 = root_path + "/" + file[1]
	folder_out = out_path + "/" + file[2]
	img1_out = folder_out + "/" + file[0]
	img2_out = folder_out + "/" + file[1]

	print img1 +"->"+img1_out
	print img2 +"->"+img2_out
	print folder_out

	try:
		os.makedirs(folder_out)
	except Exception:
		pass

	subprocess.call(["cp",img1,img1_out])
	subprocess.call(["cp",img2,img2_out])

		
def main():
	

	for i in files:
		while len(multiprocessing.active_children()) > MAX_OF_THREADS:
			pass

		p = multiprocessing.Process(target=compare,args=(i,))
		p.start()
		

main()

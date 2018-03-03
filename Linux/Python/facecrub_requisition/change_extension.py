import cv2, os, multiprocessing,sys


#the first index os argv is the name of the scrip.py
root_path = sys.argv[1] #insert root path
out_path = sys.argv[2] #insert output file
new_ext = sys.argv[3] #insert the new extention

print root_path
print out_path
print new_ext

sub_folders = os.listdir(root_path)

MAX_OF_THREADS = 9

def change_ext(sub_folder,img_file):

	img_name_path = root_path+"/"+sub_folder+"/"+img_file
	
	# open the img file 
	try:
		os.makedirs(out_path+"/"+sub_folder)
	except:
		pass

	try:
		img_cv = cv2.imread(img_name_path,cv2.CV_LOAD_IMAGE_COLOR)
	except:
		pass

	size = len(img_file)
	
	img_out_name = img_file[0:size-3] + new_ext

	img_out_path = out_path+"/"+sub_folder+"/"+ img_out_name
	print img_out_path
	cv2.imwrite(img_out_path,img_cv)

	
def main():
	for i in range(0,len(sub_folders)):
		img_files = os.listdir(root_path +"/"+sub_folders[i])
		for j in range(len(img_files)):
			while len(multiprocessing.active_children()) > MAX_OF_THREADS:
				pass

			p = multiprocessing.Process(target=change_ext,args=(sub_folders[i],img_files[j]),)
			p.start()
		

main()

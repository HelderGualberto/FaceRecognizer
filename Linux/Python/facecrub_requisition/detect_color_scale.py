import os, cv2, numpy as np, subprocess,multiprocessing

cascade_path = "/home/user/opencv-2.4.12/data/haarcascades/haarcascade_frontalface_alt.xml"
root_path = "./Images/Actors"
out_path = "./Images/Actors_gray"
sub_folders = os.listdir(root_path)
facecascade = cv2.CascadeClassifier(cascade_path)
log_file = open("./log_files/face_detect_log.txt","wr")
MAX_OF_THREADS = 9

def detect_color(img_cv,img_name):
	h,w,c = img_cv.shape
	size = h/2
	
	RGB_d = 0
	for i in range(0,size):
		R = img_cv[i][i][0]
		G = img_cv[i][i][1]
		B = img_cv[i][i][2]
		
		if R != G or R != B or G != B:
			RGB_d += 1

	if RGB_d > h/10:
		return 0	

	print img_name, RGB_d
	return 1

def move_gray_img(sub_folder,img_file):
	sub_folder_path = out_path +"/"+sub_folder
	

	img_name_out = out_path + "/"+sub_folder+"/"+img_file
	isnt_gray = 0
	img_name_path = root_path+"/"+sub_folder+"/"+img_file
	
	# open the img file 
		
	try:
		img_cv = cv2.imread(img_name_path,cv2.CV_LOAD_IMAGE_COLOR)
	except Exception:
		log_file.write("Error opening image: "+img_name_path)
		pass

	r = detect_color(img_cv,img_name_path)

	if r:
		try:
			os.makedirs(sub_folder_path)
		except Exception:
			pass
		subprocess.call(["mv",img_name_path,img_name_out])




def main():
	for i in range(0,len(sub_folders)):
		img_files = os.listdir(root_path +"/"+sub_folders[i])
		for j in range(len(img_files)):
			while len(multiprocessing.active_children()) > MAX_OF_THREADS:
				pass

			p = multiprocessing.Process(target=move_gray_img,args=(sub_folders[i],img_files[j]),)
			p.start()

main()
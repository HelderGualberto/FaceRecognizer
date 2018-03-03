import urllib, io, os,cv2, numpy, math
import thread, multiprocessing, time, sys
import numpy as np
###--------------------------------- GLOBAL VARIABLES ---------------------###

haarcascade_path = "/home/user/opencv-2.4.12/data/haarcascades/haarcascade_frontalface_alt.xml"
facecascade = cv2.CascadeClassifier(haarcascade_path)

if(sys.argv[1] == "woman"):
	file_path = "./facescrub_actresses.txt"
	error_file_path = "./error_log_actresses.txt"
else:
	file_path = "./facescrub_actors.txt"
	error_file_path = "./error_log_actors.txt"

print sys.argv[1]


def get_std_face(img_cv,ref_rect):
	img_gray = cv2.cvtColor(img_cv,cv2.COLOR_BGR2GRAY)

	try:
		#detectMultiscale is an general function to recognize objects, depends on the selected cascade
		faces = facecascade.detectMultiScale(
			img_gray, #grays caled image
			scaleFactor=1.1, # scale of the image
			minNeighbors=5, #number of faces detected in a range
			minSize=(30,30), #min size of an image
			flags=cv2.cv.CV_HAAR_SCALE_IMAGE
		)
	except Exception:
		raise ValueError("Error while searching faces")

	if len(faces)!=0:

		# to get the tuple in a array we only need to use a simple variable
		# to get the tuple elements we need to use the variable with parentheses like (a)

		if len(faces) > 1:
			face_rect = find_right_face(faces,ref_rect)
			img_out = resize_img(face_rect,img_cv)
			 # crop the image ata the poins x0,y0 and x,y
		else:
			img_out = resize_img(faces[0],img_cv)

		#return the final image with crop and resize applied
		return img_out

	else:
		raise ValueError("No faces founded")
		

###--------------------------------- FUNCTION RESIZE AND CROP IMAGE ---------------------###

def resize_img(faces_rect,img_cv):
#	cv2.rectangle(img_cv,(x,y),(x+w,h+y),(0,255,0),2) - Draw a rectangle in the specified rect
	x = faces_rect[0]
	y = faces_rect[1]
	w = faces_rect[2]
	h = faces_rect[3]

	im_height = img_cv.shape[0] # shape index 0 shows the image height value
	im_width = img_cv.shape[1] # shape index 0 shows the image height value

	if x > w/4: #that means the resultant X value is smaller than the old x
		faces_rect[0] = x - (w/8)
	else:
		faces_rect[0] = 0

	if y > h/4: #that means the resultant Y value is smaller than the old y
		faces_rect[1] = y - (h/8)
	else:
		faces_rect[1] = 0

	if im_height > h*5/4:
		faces_rect[2] = h*5/4
	else:
		faces_rect[2] = im_height

	if im_width > w*5/4:
		faces_rect[3] = w*5/4
	else:
		faces_rect[3] = im_width

	x = faces_rect[0]
	y = faces_rect[1]
	w = faces_rect[2]
	h = faces_rect[3]

	crop_img = img_cv[y:y+h,x:x+w]

	return crop_img

###--------------------------------- FUNCTION FIND RIGHT FACE ---------------------###

def find_right_face(faces,ref_rect):
	rect = (0,0,0,0,)
	dist = 9999
	for face_rect in faces:
		aux = get_face_distance(face_rect,ref_rect)
		if  aux < dist:
			rect = face_rect
			dist = aux
	return rect

###--------------------------------- FUNCTION GET FACE DISTANCE ---------------------###

def get_face_distance(face_rect,ref_rect):
	ref_x = ref_rect[0]
	ref_y = ref_rect[1]

	x = face_rect[0]
	y = face_rect[1]
	w = face_rect[2]
	h = face_rect[3]

	Xc = (x) - (w/2)
	Yc = (y) - (h/2)

	dist = math.sqrt(math.pow((ref_y - Yc),2) + math.pow((ref_x - Xc),2))
	return dist

###------------------------------BUILD STANDARD PATH TO SAVE IMAGES--------------------###

def build_std_path(name,_id):
    size = len(_id)
    path = _id
    
    while size < 5:
        path = "0" + path
        size += 1
    return name + "-" + path

###--------------------------------GET IMAGE FROM URL FUNCTION--------------------------###

def get_img_from_URL(URL,ID,img_path,rect_array):
	error_log = open(error_file_path,"a")
	
	try:
		#open the URL and dowload the data
		img_req = urllib.urlopen(URL)
		try:
			#convert the data into a array with the opencv standard
			img_bin = np.asarray(bytearray(img_req.read()),dtype=np.uint8)
			try:
				#transform the RGB standard opencv
				img_cv = cv2.imdecode(img_bin,-1)		
				try:
					#get the faces on the image
					img_cv = get_std_face(img_cv,rect_array)
					try:
						#save the image file as a jpg file
						cv2.imwrite(img_path,img_cv)
					except Exception:
						error_log.write("Error saving Image ID: " + build_std_path(" ",ID) + "\n")
				except ValueError,e:
					error_log.write( e + build_std_path(" ",ID) + "\n")
			except Exception:
				error_log.write("Error converting RGB" + build_std_path(" ",ID) + "\n")
		except Exception:
			error_log.write("Error converting numpy array" + build_std_path(" ",ID) + "\n")
	except Exception:
		error_log.write("Error opening URL img ID: " + build_std_path(" ",ID) + "\n")

	error_log.close()

##----------------------------PROCESS CLASS--------------------------------------###
class my_process(multiprocessing.Process):

    def kill_process(self,delay):
        time.sleep(delay)
        self.terminate()

    def time_out(self,delay):
        thread.start_new_thread(self.kill_process,(delay,))

###-------------------------------MAIN FUNCTION-----------------------------------###

def main():
	### ----variables declaration-----###
	delay = 180 #max time to dowload an image in seconds
	counter = 0
	name_index = 0
	URL_index = 3
	ID_index = 2
	FACE_rect_index = 4
	MAX_OF_THREADS = 8
	haarcascade_path = "/home/user/opencv-2.4.12/data/haarcascades/haarcascade_frontalface_alt.xml"
	facecascade = cv2.CascadeClassifier(haarcascade_path)
	compare_name = "initial"

	

	#Count the number of lines in the data_file
	file = open(file_path,"r")
	number_of_lines = sum(1 for lines in file)
	print int(number_of_lines)

	#open the data_file
	file = open(file_path,"r") 
	#Clean the log_error file and starts a new one
	error_log = open(error_file_path,"w")
	error_log.close()

	line = file.readline() # consome o cabecalho do csv

	while True:

		line = file.readline()
		
		if line == '':
			break

		splited_line = line.split("\t")
		name = splited_line[name_index]
		URL = splited_line[URL_index]
		ID = splited_line[ID_index]
		FACE_rect = splited_line[FACE_rect_index]
		rect_array = FACE_rect.split(',')
		
		if name != compare_name:
			compare_name = name
			try:
				os.makedirs("./images/" + name)
			except:
				Exception

		img_name = build_std_path(name,ID)
		img_path = "./images/" + compare_name +"/"+ img_name + ".jpg"

		if __name__ == '__main__':
			while len(multiprocessing.active_children()) > MAX_OF_THREADS:
				pass

			p = my_process(target=get_img_from_URL,args=(URL,ID,img_path,rect_array,))
			p.start()

			p.time_out(delay)
		else:
			print "Process ERROR!"

		#update the progress status
		progress = (int(counter)*100)/int(number_of_lines)
		print str(progress) + "%"
		counter += 1

	print "100%"

	file.close()

	return

######################################CALL THE MAIN FUNCTION #####################################

main()
import urllib, io, Image, os
import multiprocessing
import time
import thread

delay = 10 #max time to dowload an image in seconds
counter = 0
name_index = 0
URL_index = 3
ID_index = 2
FACE_rect_index = 4

LEFT_i = 0
TOP_i = 1
RIGHT_i = 2
BOTTOM_i = 3

MAX_OF_THREADS = 4

compare_name = "initial"

def build_std_path(name,_id):
    size = len(_id)
    path = _id
    
    while size < 5:
        path = "0" + path
        size += 1
    return name + "-" + path

def crop_img(img,rect_array):
	left = int(rect_array[LEFT_i])
	top = int(rect_array[TOP_i])
	right = int(rect_array[RIGHT_i])
	bottom = int(rect_array[BOTTOM_i])
	c_img = img.crop((left,top,right,bottom))
	return c_img

def get_img_from_URL(URL,ID,img_path,rect_array):
	error_log = open("d:/log.txt","a")
	
	try:
		url_img = urllib.urlopen(URL)
		try:
			img_file = io.BytesIO(url_img.read())
			img_out = Image.open(img_file)
			print img_out
			try:
				img_out = crop_img(img_out,rect_array)
				try:
					img_out.save(img_path)
				except Exception:
					error_log.write("Error saving Image ID: " + build_std_path(" ",ID) + "\n")
			except Exception:
				error_log.write("Error croping Image ID: " + build_std_path(" ",ID) + "\n")
		except Exception:
			error_log.write("Error converting Image ID: " + build_std_path(" ",ID) + "\n")		
	except Exception:
		error_log.write("Error opening URL img ID: " + build_std_path(" ",ID) + "\n")
		print "teste"
	
	error_log.close()


class my_process(multiprocessing.Process):

    def kill_process(self,delay):
        time.sleep(delay)
        self.terminate()

    def time_out(self,delay):
        thread.start_new_thread(self.kill_process,(delay,))


# Main script
file = open("d:/facescrub_actors.txt","r")
number_of_lines = sum(1 for lines in file)
print int(number_of_lines)

file = open("d:/facescrub_actors.txt","r")
error_log = open("d:/error_log.txt","w")
error_log.close()


line = file.readline() # consome o cabecalho do csv

while True:
	progress = (int(counter)*100)/int(number_of_lines)
	#print str(progress) + "%"

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
			os.makedirs("images/" + name)
		except:
			Exception

	img_name = build_std_path(name,ID)
	img_path = "images/" + compare_name +"/"+ img_name + ".jpg"

	if __name__ == '__main__':
		while len(multiprocessing.active_children()) > MAX_OF_THREADS:
			pass

		p = my_process(target=get_img_from_URL,args=(URL,ID,img_path,rect_array,))
		p.start()

		p.time_out(delay)
	else:
		print "Process ERROR!"
	counter += 1

print "100%"

file.close()
import Image, urllib, io

def crop_img(img,rect_array):
	left = int(rect_array[LEFT_i])
	top = int(rect_array[TOP_i])
	right = int(rect_array[RIGHT_i])
	bottom = int(rect_array[BOTTOM_i])
	c_img = img.crop((left,top,right,bottom))
	return c_img

url_img = urllib.urlopen("http://0.tqn.com/d/movies/1/0/x/E/Y/rum-diary-aaron-eckhart.jpg")
img_file = io.BytesIO(url_img.read())
img_out = Image.open(img_file)
img_out = crop_img(img_out,(87,89,233,235))

while True:
	img_out.show()
	time.sleep(1)

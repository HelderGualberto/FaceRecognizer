import os, subprocess

path = "./Images/Actresses/Joanna Garcia"

files = os.listdir(path)

for i in range(0,len(files)):
	file_name = files[i]
	new_name = list(file_name)
	new_name[11] = 'i'
	new_name = ''.join(new_name)
	subprocess.call[("mv",path+"/"+file_name,path+"/"+new_name),]
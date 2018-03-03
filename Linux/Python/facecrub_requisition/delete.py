import os, re, subprocess

RE = "[0-9]{5}_[0-9]{6}_(f|h|r|q)[a-z](.){5}bz2"
path = "./Images/colorferet/cd1"

lexical_analiser = re.compile(RE)
sub_folders = os.listdir(path)

for i in range(0,len(sub_folders)):
	files = os.listdir(path+"/"+sub_folders[i])

	for j in range(0,len(files)):
		file_path = path +"/"+sub_folders[i]+"/"+files[j]
		if lexical_analiser.search(files[j]):
			subprocess.call(["rm",file_path])



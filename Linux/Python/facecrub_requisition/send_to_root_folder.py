import os, sys, shutil

root_path = sys.argv[1]


folders = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]

for i in range(len(folders)):
	path = root_path + "//" + folders[i]
	files = os.listdir(path)
	try:
		for j in range(len(files)):
			shutil.move(path+"//"+files[j],root_path+"//"+files[j])
			print root_path+"//"+files[j]+" --> Moved"
	except Exception:
		pass
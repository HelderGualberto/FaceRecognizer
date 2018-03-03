#Autor: Roberto Kenji Hiramatsu
#Suporte para copia de arquivo por pscp
#Data: 2016-09-04

from subprocess import call
import os,sys

def copia(ident,desti="D:\\temp\\teste4",pastaOri="/home/yakumo/openface/huawei/usputil/video4"):
	nident = int(ident)
	caminho = "{}/*_i{:04d}_*".format(pastaOri,nident)
	orige="yakumo@localhost:{}".format(caminho)
	call(["pscp","-P","4722","-i","C:\\Users\\kenji\\face.ppk",orige,desti])

def pastaOk(desti="D:\\temp\\teste4\\"):
	if os.path.exists(desti):
		if os.path.isdir(desti):
			return True
	else:
		os.mkdir(desti)
		return True
	return False

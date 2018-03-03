#!/usr/bin/python
import os
import sys

def trataraiz(origem,destin,limsup):
    listaro = os.listdir(origem)
    idxp = 0 ;
    for subdir in listaro :
      if (os.path.isdir(origem+"/"+subdir)) :
        processa(origem+"/"+subdir,destin+"/person-"+str(idxp),limsup)
      processaprop(origem+"/"+subdir,origem+"/person-"+str(idxp))
      idxp = idxp + 1;
    return

def processa(origem,destino,limsup):
    if ( not os.path.exists(destino) ):
       print "Criando diretorio",destino
       os.mkdir(destino)
    listao = os.listdir(origem)
    narq = len(listao)
    metades = int(limsup) / 2 ;
    if ( narq > metades ) :
      idxi = 0 ;
      for idx in range(metades,narq) :
        print idx,origem+"/"+listao[idx]+" -> "+destino+"/image-"+str(idxi)+".png" ;
        os.rename(origem+"/"+listao[idx],destino+"/image-"+str(idxi)+".png")
        idxi = idxi + 1;
    return

def processaprop(origem,destino):
    if ( not os.path.exists(destino) ):
       print "Criando diretorio",destino
       os.mkdir(destino)
    lista = os.listdir(origem) ;
    idxi = 0 ;
    for arq  in lista :
      print origem+"/"+arq+" -> "+destino+"/image-"+str(idxi)+".png";
      os.rename(origem+"/"+arq,destino+"/image-"+str(idxi)+".png")
      idxi = idxi + 1 ;
    print "Removendo",origem
    os.removedirs(origem)
    return

def is_number(s) :
  try:
    int(s)
    return True
  except ValueError:
    return False

nargs = len(sys.argv)
if ( nargs > 2 ):
  origem = sys.argv[1]
  destin = sys.argv[2]
  if (nargs > 3 ):
    limsup = sys.argv[3]
  else :
    limsup = 5
  if ( nargs == 5 ):
    liminf = sys.argv[4]
  else :
    liminf = 3
  if ( os.path.exists(origem) and os.path.isdir(origem)) :
    if ( os.path.exists(destin) and os.path.isdir(destin)) :
      if( is_number(limsup) ) :
            print "    Origem: ",origem
            print "    Destino:",destin
            print "Limite sup: ",limsup
            trataraiz(origem,destin,limsup)
      else :
        print limsup," nao e numero"
    else :
      print destin," nao existe ou nao e diretorio "
  else :
    print origem," nao existe ou nao e diretorio "
else :
  print "reduze <origem> <destino> [limsup=20] [liminf=3]"

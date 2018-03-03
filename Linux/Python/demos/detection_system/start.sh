
#Este script executa o sistema de reconhecimento facial.
#Utiliza o PYTHONPATH para indicar o caminho das dependencias do sistema
#Possui flags de execucao e configuracao do sistema:

#	--video <string> : Indica a URL da camera que deseja executar o programa. Para cada --video eh obrigatoria a utilizacao do identificador da camera, atraves da flag --id

#	--id <string> : Indica o identificador respectivo a camera adicionada

#	--gammaCorrection <float> : Indica o fator de correcao gamma que se deseja aplicar em imagens com iluminacao baixa. Quando a flag nao eh configurada o sistema nao aplica correcao de gamma.

#	--useKurento : Flag que ativa a utilizacao do servidor Kurento para alertas. (Default TRUE)

#	--kurento <string> : Indica o IP de um servidor kurento externo (servidor de aplicacao)

#	--timeDelay <float> : Aplica delay entre o reconhecimento facial (tempo para o proximo reconhecimento em segundos)

#	--frameDelay <int> : Aplica delay no reconhecimento facial (numero de frames para o proximo reconhecimento)

#	--cuda : Ativa o cuda para processamento em GPU (default TRUE)
PYTHONPATH=../../usputil  python recognition_system.py \
--video rtsp://viwer:fullsec123@10.1.1.6:554/H264 --id fullsec  --gammaCorrection 2.2 \
--video rtsp://admin:huawei123@10.1.1.0:554/LiveMedia/ch1/Media1 --id Huawei --gammaCorrection 2.2
#--video "rtsp://admin:admin@10.1.1.13:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif" --id Intelbras --gammaCorrection 2.2
#--video ./data/LV_AMBIENT.mp4 --id angle --rotateIm True --gammaCorrection 2.2
#--video rtsp://viwer:foscam123@10.1.1.5:554/videoMain --id corr_sala_helder --gammaCorrection 2.2 \

#--video rtsp://viwer:foscam123@10.1.1.5:554/videoMain --id corr_sala_helder \
#--video rtsp://viwer:foscam123@10.1.1.2:554/videoMain --id corr_sala_humanlab \
#--video rtsp://viwer:foscam123@10.1.1.3:554/videoMain --id corr_principal_frente 
#--video rtsp://viwer:foscam123@10.1.1.4:554/videoMain --id corr_principal_meio \
#--video rtsp://viwer:foscam123@10.1.1.12:554/videoMain --id corr_principal_fundo \

#PYTHONPATH=../../usputil  python recognition_system.py \
#--video ./face.webm --id corr_sala_helder \
#--video ./face.webm --id corr_sala_humanlab \
#--video ./face.webm --id corr_principal_frente \
#--video ./face.webm --id corr_principal_meio \
#--video ./face.webm --id corr_principal_fundo \
#--video ./face.webm --id Huawei \
#--video ./face.webm --id fullsec 










#PYTHONPATH=../../usputil  python cam_backup.py --frameDelay 5 --video rtsp://admin:huawei123@10.1.1.0:554/LiveMedia/ch1/Media1 

#PYTHONPATH=../../usputil  python cam.py --frameDelay 10 --video "rtsp://viwer:fullsec123@10.1.1.6:554/H24"

#PYTHONPATH=../../usputil  python cam3.py --video "Bloco C - 15h30 - 16h00.mp4"

#PYTHONPATH=../../usputil python cam3.py --video "auditorio/hiv00009.mp4"
#PYTHONPATH=../../usputil  python cam3.py --video "rtsp://viwer:foscam123@10.1.1.5:554/videoMain"
#PYTHONPATH=../../usputil  python cam3.py --video rtsp://viewer:hikivision123@192.168.10.181:554/Streaming/Channels/1


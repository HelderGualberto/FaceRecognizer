# Script para extrair candidados para face nos frames com movimento
#PYTHONPATH=/vmstor/openface/SafetyCity/Code/Python/usputil/ python /vmstor/openface/SafetyCity/Code/Python/tools/extractSamp.py -i /vmstor/openface/SafetyCity/videos/sequence/p.1223084200 -s 192 >log.1223084200.txt
#PYTHONPATH=/vmstor/openface/SafetyCity/Code/Python/usputil/ python /vmstor/openface/SafetyCity/Code/Python/tools/extractSamp.py -i /vmstor/openface/SafetyCity/videos/sequence/p.1222084100 -o /vmstor/openface/SafetyCity/imagens -s 192 >log.1222084100.txt
#PYTHONPATH=/srv/safetycity/Code/Python/usputil/ python /srv/safetycity/Code/Python/tools/extractSamp.py -i /srv/imagens_filtradas -o /srv/safetycity/imagens/imagens_filtradas --enabledb False -u mongodb://mdb:27017 -s 240  >log.filtradas.txt
#PYTHONPATH=/srv/safetycity/Code/Python/usputil/ python /srv/safetycity/Code/Python/tools/extractSamp.py -i /srv/amostrab -o /srv/safetycity/imagens/amostrab --enabledb False -u mongodb://mdb:27017 -s 240  >log.filtradas.txt
#PYTHONPATH=/vmstor/openface/SafetyCity/Code/Python/util/ python /vmstor/openface/SafetyCity/Code/Python/tools/extractSamp.py -i /vmstor/openface/SafetyCity/videos/sequence/p.1225084400 -o /vmstor/openface/SafetyCity/imagens/p.1225084400 -s 240 >log.1225084400.txt
#PYTHONPATH=/vmstor/openface/SafetyCity/Code/Python/util/ python /vmstor/openface/SafetyCity/Code/Python/tools/extractSamp.py -i /vmstor/openface/SafetyCity/videos/sequence/p.1226084400 -o /vmstor/openface/SafetyCity/imagens/p.1226084400 -s 240 >log.1226084400.txt
#PYTHONPATH=/vmstor/openface/SafetyCity/Code/Python/util/ python /vmstor/openface/SafetyCity/Code/Python/tools/extractSamp.py -i /vmstor/openface/SafetyCity/videos/sequence/p.1227084500 -o /vmstor/openface/SafetyCity/imagens/p.1227084500 -s 240 >log.1227084500.txt
#for SER in 1222023104 0103193600 1222034541
# deve comerar do diretorio que contem processa.sh
cd /vmstor/openface/SafetyCity/Code/Python/tools
for SER in `python /vmstor/openface/SafetyCity/Code/Python/tools/listPExtrair.py`
do 
PYTHONPATH=/vmstor/openface/SafetyCity/Code/Python/util/ python /vmstor/openface/SafetyCity/Code/Python/tools/extractSamp.py -i /vmstor/openface/SafetyCity/videos/sequence/p.$SER -o /vmstor/openface/SafetyCity/imagens/p.${SER} -s 240 >logs/log.${SER}.txt
done

# Script para separar frames com movimento de outros
#python /iscsi/videos/bench/testbackdetA.py --video /iscsi/videos/corredor/corredor-1228084501.mp4 > log.1228084501.txt
#python /iscsi/videos/bench/testbackdetA.py --video /iscsi/videos/corredor/corredor-1229084500.mp4 > log.1229084500.txt
#python /iscsi/videos/bench/testbackdetA.py --video /iscsi/videos/corredor/corredor-1213140207.mp4 > log.1213140207.txt
#python /iscsi/videos/bench/testbackdetA.py --video /iscsi/videos/corredor/corredor-1214070400.mp4 > log.1214070400.txt
#python /iscsi/videos/bench/testbackdetA.py --video  /iscsi/videos/corredor/corredor-1231084500.mp4 > log.1231084500.txt
#python /iscsi/videos/SafetyCity/Code/Python/tools/movingdet/testbackdetA.py --video  /iscsi/videos/corredor/corredor-1214130402.mp4 > logs/log.1214130402.txt
#python /iscsi/videos/SafetyCity/Code/Python/tools/movingdet/testbackdetA.py --video  /iscsi/videos/corredor/corredor-1213140207.mp4 > logs/log.1213140207.txt
#python /iscsi/videos/SafetyCity/Code/Python/tools/movingdet/testbackdetA.py --video  /iscsi/videos/corredor/corredor-1214170708.mp4 > logs/log.1214170708.txt
#mv  /iscsi/videos/corredor/corredor-1214170708.mp4  /iscsi/videos/corredor/processado
#python /iscsi/videos/SafetyCity/Code/Python/tools/movingdet/testbackdetA.py --video  /iscsi/videos/corredor/corredor-1214180710.mp4 > logs/log.1214180710.txt
#mv  /iscsi/videos/corredor/corredor-1214180710.mp4  /iscsi/videos/corredor/processado
#python /iscsi/videos/SafetyCity/Code/Python/tools/movingdet/testbackdetA.py --video  /iscsi/videos/corredor/corredor-1214190712.mp4 > logs/log.1214190712.txt
#mv  /iscsi/videos/corredor/corredor-1214190712.mp4  /iscsi/videos/corredor/processado
#python /iscsi/videos/SafetyCity/Code/Python/tools/movingdet/testbackdetA.py --video  /iscsi/videos/corredor/corredor-1214200716.mp4 > logs/log.1214200716.txt
#mv  /iscsi/videos/corredor/corredor-1214200716.mp4  /iscsi/videos/corredor/processado
#python /iscsi/videos/SafetyCity/Code/Python/tools/movingdet/testbackdetA.py --video  /iscsi/videos/corredor/corredor-1222023104.mp4 > logs/log.1222023104.txt
#mv  /iscsi/videos/corredor/corredor-1222023104.mp4  /iscsi/videos/corredor/processado
#python /iscsi/videos/SafetyCity/Code/Python/tools/movingdet/testbackdetA.py --video  /iscsi/videos/corredor/corredor-1222034101.mp4 > logs/log.1222034101.txt
#mv  /iscsi/videos/corredor/corredor-1222034101.mp4  /iscsi/videos/corredor/processado
#python /iscsi/videos/SafetyCity/Code/Python/tools/movingdet/testbackdetA.py --video  /iscsi/videos/corredor/corredor-1222034541.mp4 > logs/log.1222034541.txt
#mv  /iscsi/videos/corredor/corredor-1222034541.mp4  /iscsi/videos/corredor/processado
#python /iscsi/videos/SafetyCity/Code/Python/tools/movingdet/testbackdetA.py --video  /iscsi/videos/corredor/video-corredor-12121201.mp4 > logs/log.12121201.txt
#mv  /iscsi/videos/corredor/video-corredor-12121201.mp4  /iscsi/videos/corredor/processado
#python /iscsi/videos/SafetyCity/Code/Python/tools/movingdet/testbackdetA.py --video  /iscsi/videos/corredor/video-corredor-12121341.mp4 > logs/log.12121341.txt
#mv  /iscsi/videos/corredor/video-corredor-12121341.mp4  /iscsi/videos/corredor/processado
#python /iscsi/videos/SafetyCity/Code/Python/tools/movingdet/testbackdetA.py --video  /iscsi/videos/corredor/video-corredor-1212.mp4 > logs/log.1212.txt
#mv  /iscsi/videos/corredor/video-corredor-1212.mp4  /iscsi/videos/corredor/processado
#python /iscsi/videos/SafetyCity/Code/Python/tools/movingdet/testbackdetA.py --video  /iscsi/videos/corredor/video-corredor-12130928.mp4 > logs/log.12130928.txt
#mv  /iscsi/videos/corredor/video-corredor-12130928.mp4  /iscsi/videos/corredor/processado
#python /iscsi/videos/SafetyCity/Code/Python/tools/movingdet/testbackdetA.py --video  /iscsi/videos/corredor/video-corredor-12131341.mp4 > logs/log.12131341.txt
#mv  /iscsi/videos/corredor/video-corredor-12131341.mp4  /iscsi/videos/corredor/processado
#python /iscsi/videos/SafetyCity/Code/Python/tools/movingdet/testbackdetA.py --video  /iscsi/videos/corredor/corredor-0104200500.mp4 > logs/log.0104200500.txt
#mv  /iscsi/videos/corredor/corredor-0104200500.mp4  /iscsi/videos/corredor/processado
#python /iscsi/videos/SafetyCity/Code/Python/tools/movingdet/testbackdetA.py --video  /iscsi/videos/corredor/corredor-0104234700.mp4 > logs/log.0104234700.txt
#mv  /iscsi/videos/corredor/corredor-0104234700.mp4  /iscsi/videos/corredor/processado
#for SER in 0109212200 0110104300
#for SER in  0111120600  0112101800
for SER in  0119074500
do
python /iscsi/videos/SafetyCity/Code/Python/tools/movingdet/testbackdetA.py --video  /iscsi/videos/corredor/corredor-${SER}.mp4 > logs/log.${SER}.txt
mv  /iscsi/videos/corredor/corredor-${SER}.mp4  /iscsi/videos/corredor/processado
done

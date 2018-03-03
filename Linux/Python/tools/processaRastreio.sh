for ITEM in `python /iscsi/videos/SafetyCity/Code/Python/tools/listPTrack.py`
do
PYTHONPATH=/iscsi/videos/SafetyCity/Code/Python/util python /iscsi/videos/SafetyCity/Code/Python/tools/markTrack2.py -b p_${ITEM}
done

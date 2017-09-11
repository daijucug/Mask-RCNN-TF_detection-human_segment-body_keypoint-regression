#!/bin/bash
out=$(ps aux | grep '/usr/bin/python /hdd1/Alex/testMaskRCNN_human_bodyparts/MaskRCNN_body/train/train.py' | rev | cut -d ' ' -f 1 | rev | wc -l)
if [ $out -eq "2" ];then
	echo "2 processes" >> /tmp/testing.txt
else
	echo "1 processes" >> /tmp/testing.txt
	echo $(date) >> /tmp/testing.txt
	export CUDA_VISIBLE_DEVICES=0
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/
	`(/usr/bin/python /hdd1/Alex/testMaskRCNN_human_bodyparts/MaskRCNN_body/train/train.py &>> /tmp/testing.txt)`
	echo "tried to start" >> /tmp/testing.txt;
fi



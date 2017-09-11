#mkdir data

############################################################################################VOC
#wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
#tar -xvzf VOCtrainval_03-May-2010.tar -C data/
#tar -xvf VOCtrainval_03-May-2010.tar -C data/
#mv  data/VOCdevkit/VOC2010/JPEGImages/ data/
#wget http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/trainval.tar.gz
#tar -xvzf trainval.tar.gz -C data/
#python convert_VOC_human_body_parts.py
#mv data/out_human_and_body_parts.tfrecord out_human_and_body_parts.tfrecord

############################################################################################Chalearn
#mkdir data/chalearn
#mkdir data/chalearn/api_code
#wget https://competitions.codalab.org/my/datasets/download/764962c6-c270-4ee1-8721-e5611a5665f2 --no-check-certificate
#wget https://competitions.codalab.org/my/datasets/download/27f9a04b-5499-4acf-b7b2-8aabb26f283c --no-check-certificate
#mv 27f9a04b-5499-4acf-b7b2-8aabb26f283c dataset.zip
#unzip dataset.zip -d data/chalearn/api_code/
#mv ChalearnLAPEvaluation.py data/chalearn/api_code/ChalearnLAPEvaluation.py
#mv ChalearnLAPSample.py data/chalearn/api_code/ChalearnLAPSample.py
#mv convert_CHALEARN_human_body_parts.py data/chalearn/api_code/convert_CHALEARN_human_body_parts.py
#cd data/chalearn/api_code
#python convert_CHALEARN_human_body_parts.py
#cd ../../..
#mv data/chalearn/api_code/out_human_and_body_parts_chalearn.tfrecord out_human_and_body_parts_chalearn.tfrecord

#############################################################################################ADE20K
#wget http://groups.csail.mit.edu/vision/datasets/ADE20K/ADE20K_2016_07_26.zip
#wget http://groups.csail.mit.edu/vision/datasets/ADE20K/code.zip
#mkdir data/ade20k
#unzip ADE20K_2016_07_26.zip -d data/ade20k/
#unzip code.zip -d data/ade20k/
#mkdir data/ade20k/output_dir
#mv data/ade20k/ADE20K_2016_07_26/index_ade20k.mat data/ade20k/index_ade20k.mat
#mv human_body_parts.m data/ade20k/human_body_parts.m
#cp data/ade20k/code/loadAde20K.m data/ade20k/loadAde20K.m
#cd data/ade20k/
#octave human_body_parts.m
#cd ../..
#python convert_ADE20k_human_body_parts.py
#mv data/out_human_and_body_parts_ade_20k_max640edge.tfrecord out_human_and_body_parts_ade_20k.tfrecord

###############################################################################################JHMDB
#wget http://files.is.tue.mpg.de/jhmdb/JHMDB_video.zip
#wget http://files.is.tue.mpg.de/jhmdb/joint_positions.zip
#wget http://files.is.tue.mpg.de/jhmdb/puppet_mask.zip
#wget http://files.is.tue.mpg.de/jhmdb/puppet_flow_com.zip
#mkdir data/jhmdb
mkdir data/jhmdb/JHMDB_video
#unzip JHMDB_video.zip -d data/jhmdb/
mv data/jhmdb/ReCompress_Videos/ data/jhmdb/JHMDB_video
#unzip joint_positions.zip -d data/jhmdb/
#unzip puppet_mask.zip -d data/jhmdb/
#unzip puppet_flow_com.zip -d data/jhmdb/

#mv convert_jhmdb.py data/jhmdb/convert_jhmdb.py
#mv read_my_data_keypoints.py data/jhmdb/read_my_data_keypoints.py
#cd data/jhmdb/
#python convert_jhmdb.py
#cd ../..
#mv data/jhmdb/out_human_and_body_parts_keypoints_JHMDB.tfrecord out_human_and_body_parts_keypoints_JHMDB.tfrecord

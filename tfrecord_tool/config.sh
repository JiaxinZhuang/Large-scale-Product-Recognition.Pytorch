export data_list_file=./data_list.txt  
#export train_data_dir=/dockerdata/fufuyu/wanda_reid_data/
export train_data_dir=""
export indices_dir=./indexes  
export remote_tfrecord_dir=/dockerdata/miyozhang/tfrecords/  
export new_index_dir=/dockerdata/miyozhang/tfrecords_txt/  


export shard_size=500000
export limit=-1
export shuffle=1

#export LD_LIBRARY_PATH=/data1/rapidflow/env/cuda-9.1/lib64:/data1/rapidflow/env/nccl_2.1.15-1+cuda9.1_x86_64/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data1/public/runtime/cuda-9.0/lib64/:/data1/public/runtime/cuda-10.0/lib64/:"$LD_LIBRARY_PATH"
export PYTHON=/dockerdata/miyozhang/anaconda3/bin/python
#export PYTHON=~/anaconda2/bin/python

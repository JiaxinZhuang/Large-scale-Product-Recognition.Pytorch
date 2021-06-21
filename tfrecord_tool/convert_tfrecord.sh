#!/usr/bin/env bash

source ./config.sh

for dataset_name in $(<$data_list_file); do
    input_index=${indices_dir}/index_${dataset_name}.txt  
    output_name=TFR-${dataset_name}
    output_dir=${remote_tfrecord_dir}/TFR-${dataset_name}

    mkdir -p $output_dir logs
    CUDA_VISIBLE_DEVICES="" 
    $PYTHON -u wanda_encode.py --input_index=${input_index} \
                      --input_dir=${train_data_dir} \
                      --output_name=${output_name} \
                      --output_dir=${output_dir} \
                      --shard_size=${shard_size} \
                      --limit=${limit} \
                      --shuffle=${shuffle} #> logs/encode_${output_name}.log 2>&1 #&
done

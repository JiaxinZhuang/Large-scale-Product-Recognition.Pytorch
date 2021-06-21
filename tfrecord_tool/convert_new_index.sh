#!/usr/bin/env bash

source ./config.sh

for dataset_name in $(<$data_list_file); do
    input_index=${indices_dir}/index_${dataset_name}.txt  # 这边的文件名如果有前缀自己加一下，比如 ReID 的 index_
    output_name=TFR-${dataset_name}
    output_dir=${remote_tfrecord_dir}/TFR-${dataset_name}

    mkdir -p $output_dir logs

    $PYTHON -u wanda_convert_new_index.py ${input_index} ${output_dir}/TFR-${dataset_name}.index TFR-${dataset_name}.txt
    shuf TFR-${dataset_name}.txt -o ${new_index_dir}/TFR-${dataset_name}.txt
    rm TFR-${dataset_name}.txt
done

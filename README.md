# Large-scale Product Recognition (CVPR 2021 AliProducts Challenge)
This repository contains source codes of SYSU-Youtu for CVPR 2021 AliProducts Challenge: Large-scale Product Recognition

You can find details from our technical reports [Solution for Large-scale Long-tailed Recognition with Noisy Labels](https://trax-geometry.s3.amazonaws.com/cvpr_challenge/cvpr2021/recognition_challenge_technical_reports/2nd+Place+Solution+to+CVPR+2021+AliProducts+Challenge.pdf).

Our solution obtains 6.4365% mean class error rate in the leaderboard with our ensemble model, which ranks the Second.



## Requirements

* Pytorch 1.6+
* Cuda 10



## Prerequisite

Using code in **tfrecord_tool** to convert images from jpg to TF Records and txt.

## Run

For transformers

```
cd Deit

cp -r /path/to/TFR-aliproduct_train /dev/shm/
cp -r /path/to/TFR-aliproduct_val /dev/shm/
cp -r /path/to/*.txt /dev/shm/

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_small_patch16_224 --version _v9 --train --lr 5e-5 --epochs 100 2>&1 | tee ./logs/std.log
```



## TFRecord

[V4](https://drive.google.com/file/d/194bmtyOeZ39-EbE_jmAs78wDtF0EeWFN/view?usp=sharing)

[V9](https://drive.google.com/file/d/1Ad8s5ytnLLIVWIjZQSmGveb39T9rQpzJ/view?usp=sharing)

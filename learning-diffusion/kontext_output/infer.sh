#!/bin/bash
# run_all_gpus.sh

for i in {0..7}
do
    echo "启动 GPU $i"
    CUDA_VISIBLE_DEVICES=$i python inference.py --part $i &
done

wait
echo "所有GPU任务完成"

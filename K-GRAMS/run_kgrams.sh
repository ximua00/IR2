#!/bin/sh
CMD="srun --gres=gpu:1 -p fatq python train.py --epochs 2000 --save_freq 200"

declare -a LR=("0.001" "0.0001" "0.00003" "0.01")

for lr in "${LR[@]}"; do
    FULL="$CMD --narre_learning_rate $lr --exp_name $lr"
    echo $FULL
    eval $FULL
done

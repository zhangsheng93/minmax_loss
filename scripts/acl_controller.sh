#!/bin/bash

if [ -n "$1" ]; then
    echo "phi=$1"
else
    echo "no first parameter"
    exit 1
fi

if [ -n "$2" ]; then
    echo "epoch=$2"
    epoch_num=$2
else
    echo "no second parameter"
    epoch_num=5
fi

if [ -n "$3" ]; then
    echo "concurrent_cnt=$3"
    concurrent_cnt=$3
else
    echo "no third parameter"
    concurrent_cnt=2
fi

prefix="acl_controller"
controller="acl"
BATCH_SIZE=8
batch_size_train=8
tstr=$(date +"%F_%H%M%S")

# train_datasets="cola,sst,mrpc,stsb,qqp,mnli,qnli,rte"
# test_datasets="cola,sst,mrpc,stsb,qqp,mnli_matched,mnli_mismatched,mnli_ax,qnli,rte"
train_datasets="cola,rte"
test_datasets="cola,rte"


MODEL_ROOT="checkpoints"
BERT_PATH="mt_dnn_models/bert_model_base_uncased.pt"
DATA_DIR="data/canonical_data/bert_base_uncased_lower"

answer_opt=1
optim="adamax"
grad_clipping=0
global_grad_clipping=1.0
grad_accumulation_step=4
lr="5e-5"
max_queue_cnt=50
phi=$1
epochs=$epoch_num
log_per_updates=3000

model_dir="checkpoints/${prefix}_${optim}_phi$1_k${concurrent_cnt}_qlen${max_queue_cnt}_${tstr}"
log_file="${model_dir}/log.log"
python3 train.py --data_dir ${DATA_DIR} --init_checkpoint ${BERT_PATH} --batch_size ${BATCH_SIZE} --batch_size_train ${batch_size_train} --controller ${controller} --phi ${phi} --epochs ${epochs}  --output_dir ${model_dir} --tensorboard --log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} --train_datasets ${train_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --grad_accumulation_step ${grad_accumulation_step} --global_grad_clipping ${global_grad_clipping} --learning_rate ${lr} --multi_gpu_on --log_per_updates ${log_per_updates} --concurrent_cnt ${concurrent_cnt} --max_queue_cnt ${max_queue_cnt}
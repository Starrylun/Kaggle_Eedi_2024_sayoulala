#!/bin/bash


PATH_PRE="./"

VERSION="recall_v9_gen_step2_all_72b"
DATA_DIR=${PATH_PRE}/train_data/${VERSION}/
MODEL_USE="simcse_qwen25_32b"
ZERO_STAGE=2
OUTPUT=${PATH_PRE}/model_save/${MODEL_USE}_${VERSION}_5e-5



#模型地址
 MODEL_PATH="./Qwen2___5-72B-Instruct"
# large 114
# qwen14b 442
mkdir -p ${OUTPUT}
echo ${ZERO_STAGE}
echo ${OUTPUT}

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
echo ${MASTER_PORT}
deepspeed  --master_port ${MASTER_PORT}  --include localhost:0,1,2,3,4,5,6,7 qwen2_qlora_v1.py \
       --project_name ${name}_${MODEL_USE} \
       --train_data ${DATA_DIR}train.jsonl \
       --doc_data ${PATH_PRE}/data/misconception_mapping.csv \
       --lora_path "none"  \
       --model_name_or_path ${MODEL_PATH} \
       --per_device_train_batch_size 1 \
       --per_device_eval_batch_size 1 \
       --train_group_size 4 \
       --gradient_accumulation_steps 4 \
       --query_max_len 300 \
       --passage_max_len 300 \
       --earystop 0 \
       --save_batch_steps 100000000000 \
       --eary_stop_epoch 5 \
       --save_per_epoch 1 \
       --num_train_epochs 5  \
       --learning_rate 5e-5 \
       --num_warmup_steps 100 \
       --weight_decay 0.01 \
       --lr_scheduler_type cosine \
       --seed 1234 \
       --zero_stage $ZERO_STAGE \
       --deepspeed \
       --output_dir $OUTPUT \
       --gradient_checkpointing
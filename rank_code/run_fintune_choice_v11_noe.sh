#!/bin/bash


PATH_PRE=".temp"
MODEL_USE="rank_choice_v11"
VERSION="v9_fold_4_rank_and_warmup"
DATA_DIR=${PATH_PRE}/train_data/${VERSION}/

lora_path="none"
MODEL_PATH=./Qwen2___5-32B-Instruct-AWQ

model_name=ce
ZERO_STAGE=2
OUTPUT=${PATH_PRE}/model_save/${MODEL_USE}_${VERSION}_warmup_train_5e-5
mkdir -p ${OUTPUT}
echo ${ZERO_STAGE}
echo ${OUTPUT}
MASTER_PORT=24345
echo ${MASTER_PORT}
deepspeed  --master_port ${MASTER_PORT}  --include localhost:0,1,2,3,4,5,6,7 deepspeed_rank_choice_v11_noe.py \
       --project_name ${name}_${MODEL_USE} \
       --lora_path ${lora_path} \
       --model_name ${model_name} \
       --train_dataset_path ${DATA_DIR}train.parquet \
       --dev_dataset_path  ${DATA_DIR}dev.parquet \
       --model_name_or_path ${MODEL_PATH} \
       --use_4bit 0 \
       --per_device_train_batch_size 1 \
       --per_device_eval_batch_size 1 \
       --gradient_accumulation_steps 4 \
       --max_prompt_len 1024 \
       --max_completion_len 256 \
       --earystop 0 \
       --save_batch_steps 400 \
       --eary_stop_epoch 1000 \
       --save_per_epoch 0 \
       --num_train_epochs 5  \
       --debug_code 0 \
       --learning_rate 5e-5 \
       --num_warmup_steps 100 \
       --weight_decay 0. \
       --lr_scheduler_type cosine \
       --seed 1234 \
       --zero_stage $ZERO_STAGE \
       --deepspeed \
       --output_dir $OUTPUT \
       --gradient_checkpointing
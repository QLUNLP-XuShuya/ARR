#!/bin/bash

dataset="abn"
annotation="/mimic-cxr-jpg-2.1.0/priorrg_mimic_abn_annotation.json"
base_dir="/mimic-cxr-jpg-2.1.0/files/"
version="abn_FQR+CPRA"
delta_file=""
savepath="/ABLATION/test/$dataset/$version"

mkdir -p ${savepath}

python -u train.py \
    --test \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 8 \
    --val_batch_size 8 \
    --freeze_vm True \
    --vis_use_lora True \
    --savedmodel_path ${savepath} \
    --delta_file ${delta_file} \
    --max_length 160 \
    --min_new_tokens 40 \
    --max_new_tokens 160 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 8 \
    --devices 2 \
    --max_epochs 1 \
    --strategy ddp_find_unused_parameters_true \
    --use_cross_attention \
    --num_finding_queries 20 \
    --finding_query_dim 256 \
    --cross_attention_heads 8 \
    --cross_attention_layers 2 \
    --finding_loss_weight 0.2 \
    --use_alignment \
    --align_loss_weight 0.02 \
    --use_cot False \
    2>&1 | tee -a ${savepath}/log.txt


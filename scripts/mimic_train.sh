#!/bin/bash
dataset="mimic_cxr"
annotation="/mimic-cxr-jpg-2.1.0/mimic_annotation_all.json"
base_dir="/mimic-cxr-jpg-2.1.0/files/"
version="mimic_all"
savepath="/save/$dataset/$version"


mkdir -p ${savepath}


python -u train.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 4 \
    --val_batch_size 8 \
    --accumulate_grad_batches 4 \
    --freeze_vm True \
    --vis_use_lora True \
    --gradient_checkpointing True \
    --savedmodel_path ${savepath} \
    --max_length 160 \
    --min_new_tokens 40 \
    --max_new_tokens 160 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 8 \
    --devices 4 \
    --max_epochs 30 \
    --strategy ddp_find_unused_parameters_true \
    --precision bf16-mixed \
    --use_cross_attention \
    --num_finding_queries 8 \
    --finding_query_dim 256 \
    --cross_attention_heads 8 \
    --cross_attention_layers 2 \
    --finding_loss_weight 0.2 \
    --use_alignment \
    --align_loss_weight 0.02 \
    --use_cot \
    --cot_findings_loss_weight 0.7 \
    --cot_reasoning_loss_weight 0.3 \
    --cot_report_loss_weight 0.7 \
    --cot_max_findings_tokens 80 \
    --cot_max_reasoning_tokens 50 \
    --cot_split_strategy heuristic \
    --cot_inference_mode \
    2>&1 | tee -a ${savepath}/log.txt


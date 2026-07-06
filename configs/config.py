import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser(description="hyper-parameter for R2GenGPT")
# ========================= Dataset =========================
parser.add_argument('--test', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--validate', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--dataset', type=str, default='mimic_cxr')
parser.add_argument('--annotation', type=str, default=r'./data/mimic_cxr/annotation.json')
parser.add_argument('--base_dir', type=str, default=r'./data/mimic_cxr/images')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--val_batch_size', default=4, type=int)
parser.add_argument('--test_batch_size', default=4, type=int)
parser.add_argument('--prefetch_factor', default=4, type=int)
parser.add_argument('--num_workers', default=8, type=int)

# ========================= Model =========================
parser.add_argument('--vision_model', default='/models--microsoft--swin-base-patch4-window7-224', type=str)
parser.add_argument('--llama_model', default='/Llama-2-7b-chat-hf', type=str)
parser.add_argument('--freeze_vm', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--llm_use_lora', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--llm_r', default=16, type=int)
parser.add_argument('--llm_alpha', default=16, type=int)
parser.add_argument('--vis_use_lora', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--vis_r', default=16, type=int)
parser.add_argument('--vis_alpha', default=16, type=int)
parser.add_argument('--lora_dropout', default=0.1, type=float)
parser.add_argument('--global_only', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--low_resource', default=False, type=bool)
parser.add_argument('--end_sym', default='</s>', type=str)

# ========================= SavedModel =========================
parser.add_argument('--savedmodel_path', type=str, default='save/mimic/v1')
parser.add_argument('--ckpt_file', type=str, default=None)
parser.add_argument('--delta_file', type=str, default=None)
parser.add_argument('--weights', type=list, default=[0.5, 0.5])
parser.add_argument('--scorer_types', type=list, default=['Bleu_4', 'CIDEr'])

# ========================= Learning =========================
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--gradient_clip_val', default=1.0, type=float)
parser.add_argument('--gradient_clip_algorithm', type=str, default='norm', choices=['norm', 'value'])
parser.add_argument('--gradient_checkpointing', default=True,
                    type=lambda x: (str(x).lower() == 'true'),
                    help='Enable gradient checkpointing on LLaMA. ~30-50% activation memory saved.')

# ========================= Decoding =========================
parser.add_argument('--beam_size', type=int, default=3)
parser.add_argument('--do_sample', type=bool, default=False)
parser.add_argument('--no_repeat_ngram_size', type=int, default=2)
parser.add_argument('--num_beam_groups', type=int, default=1)
parser.add_argument('--min_new_tokens', type=int, default=80)
parser.add_argument('--max_new_tokens', type=int, default=120)
parser.add_argument('--max_length', type=int, default=100)
parser.add_argument('--repetition_penalty', type=float, default=2.0)
parser.add_argument('--length_penalty', type=float, default=2.0)
parser.add_argument('--diversity_penalty', type=float, default=0)
parser.add_argument('--temperature', type=float, default=0)

# ========================= PL =========================
parser.add_argument('--devices', type=int, default=2)
parser.add_argument('--num_nodes', type=int, default=1)
parser.add_argument('--accelerator', type=str, default="gpu")
parser.add_argument('--strategy', type=str, default="ddp")
parser.add_argument('--precision', type=str, default='bf16-mixed')
parser.add_argument('--limit_val_batches', type=float, default=1.0)
parser.add_argument('--limit_test_batches', type=float, default=1.0)
parser.add_argument('--limit_train_batches', type=float, default=1.0)
parser.add_argument('--max_epochs', type=int, default=3)
parser.add_argument('--every_n_train_steps', type=int, default=0)
parser.add_argument('--val_check_interval', type=float, default=1.0)
parser.add_argument('--accumulate_grad_batches', type=int, default=4)
parser.add_argument("--num_sanity_val_steps", type=int, default=2)

# =========================Alignment=========================
parser.add_argument('--use_alignment', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--extract_phrases_for_alignment', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--cache_alignment', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--max_phrases_per_report', type=int, default=15)
parser.add_argument('--align_loss_weight', type=float, default=0.1)
parser.add_argument('--align_temperature', type=float, default=0.07)
parser.add_argument('--align_topk', type=int, default=5)
parser.add_argument('--use_medical_encoder', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--medical_encoder_model', type=str, default='emilyalsentzer/Bio_ClinicalBERT')

# =========================Cross-Attention=========================
parser.add_argument('--use_cross_attention', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--num_finding_queries', type=int, default=20)
parser.add_argument('--finding_query_dim', type=int, default=256)
parser.add_argument('--cross_attention_heads', type=int, default=8)
parser.add_argument('--cross_attention_layers', type=int, default=2)
parser.add_argument('--finding_loss_weight', type=float, default=0.1)
parser.add_argument('--finding_categories', type=list, default=[
    'effusion', 'atelectasis', 'opacity', 'nodule', 'mass',
    'consolidation', 'infiltrate', 'pneumonia', 'edema',
    'pneumothorax', 'cardiomegaly', 'calcification', 'fibrosis',
    'pleural_thickening', 'granuloma', 'scarring', 'emphysema',
    'cavitation', 'bronchiectasis', 'hilum_adenopathy'
])

# =========================Reasoning=========================
parser.add_argument('--use_cot', type=str2bool, nargs='?', const=True, default=True,
                    help='Enable single-pass CoT multi-head decoder')
parser.add_argument('--cot_findings_loss_weight', type=float, default=0.3)
parser.add_argument('--cot_reasoning_loss_weight', type=float, default=0.3)
parser.add_argument('--cot_report_loss_weight', type=float, default=1.0)
parser.add_argument('--cot_inference_mode', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--cot_max_findings_tokens', type=int, default=60)
parser.add_argument('--cot_max_reasoning_tokens', type=int, default=40)
parser.add_argument('--cot_split_strategy', type=str, default='heuristic',
                    choices=['heuristic', 'ratio', 'duplicate'])
parser.add_argument('--cot_split_ratio', type=float, default=0.7)

# =========================ABN (priorrg_mimic_abn) =========================
parser.add_argument('--use_prior', type=str2bool, nargs='?', const=True, default=False,
                    help='Parse prior_study (latest_study + second_recent_study) for ABN dataset.')
parser.add_argument('--use_second_recent_prior', type=str2bool, nargs='?', const=True, default=False,
                    help='Also load second_recent_study (~35%% of samples). Off => latest only.')
parser.add_argument('--max_prior_images', type=int, default=2,
                    help='Cap on prior images per sample (latest first, then second_recent).')
parser.add_argument('--use_factual_serialization', type=str2bool, nargs='?', const=True, default=True,
                    help='Use annotation-provided findings_factual_serialization as alignment phrases for ABN.')
parser.add_argument('--use_abn_meta_context', type=str2bool, nargs='?', const=True, default=False,
                    help='Prepend indication / comparison meta to raw report for ABN.')
parser.add_argument('--merge_prior_phrases', type=str2bool, nargs='?', const=True, default=False,
                    help='Merge prior factual_serialization into alignment phrase pool.')

# ========================= Misc =========================
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--warmup_steps', type=int, default=1000)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--debug', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--log_every_n_steps', type=int, default=50)
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--find_unused_parameters', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--sync_batchnorm', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--experiment_name', type=str, default=None)
parser.add_argument('--wandb_project', type=str, default='R2GenGPT')
parser.add_argument('--wandb_entity', type=str, default=None)
parser.add_argument('--wandb_offline', type=str2bool, nargs='?', const=True, default=False)

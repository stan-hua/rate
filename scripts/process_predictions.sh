#!/bin/bash
# Usage: ./process_predictions.sh <output_path> [--save_gt] <input_path>

#### TODO: change the paths

python hash_and_save.py \
    --save_name "/mnt/nfs/users/zwcolin/pillar_vlm_parent_vbc/lib/LLaMA-Factory/outputs/eval/gt.json" \
    --save_gt \
    --input_path "/mnt/nfs/users/zwcolin/pillar_vlm_parent_vbc/lib/LLaMA-Factory/outputs/eval/8b_stage1/generated_predictions.jsonl"

python hash_and_save.py \
    --save_name "/mnt/nfs/users/zwcolin/pillar_vlm_parent_vbc/lib/LLaMA-Factory/outputs/eval/8b_stage1_predictions.json" \
    --input_path "/mnt/nfs/users/zwcolin/pillar_vlm_parent_vbc/lib/LLaMA-Factory/outputs/eval/8b_stage1/generated_predictions.jsonl"

python hash_and_save.py \
    --save_name "/mnt/nfs/users/zwcolin/pillar_vlm_parent_vbc/lib/LLaMA-Factory/outputs/eval/8b_stage2_predictions.json" \
    --input_path "/mnt/nfs/users/zwcolin/pillar_vlm_parent_vbc/lib/LLaMA-Factory/outputs/eval/8b_stage2/generated_predictions.jsonl"

python hash_and_save.py \
    --save_name "/mnt/nfs/users/zwcolin/pillar_vlm_parent_vbc/lib/LLaMA-Factory/outputs/eval/32b_stage1_predictions.json" \
    --input_path "/mnt/nfs/users/zwcolin/pillar_vlm_parent_vbc/lib/LLaMA-Factory/outputs/eval/32b_stage1/generated_predictions.jsonl"

python hash_and_save.py \
    --save_name "/mnt/nfs/users/zwcolin/pillar_vlm_parent_vbc/lib/LLaMA-Factory/outputs/eval/32b_stage2_predictions.json" \
    --input_path "/mnt/nfs/users/zwcolin/pillar_vlm_parent_vbc/lib/LLaMA-Factory/outputs/eval/32b_stage2/generated_predictions.jsonl"

python hash_and_save.py \
    --save_name "/mnt/nfs/users/zwcolin/pillar_vlm_parent_vbc/lib/LLaMA-Factory/outputs/eval/8b_stage2_hightemp.json" \
    --input_path "/mnt/nfs/users/zwcolin/pillar_vlm_parent_vbc/lib/LLaMA-Factory/outputs/eval/8b_stage2_hightemp/generated_predictions.jsonl"

python hash_and_save.py \
    --save_name "/mnt/nfs/users/zwcolin/pillar_vlm_parent_vbc/lib/LLaMA-Factory/outputs/eval/32b_stage2_hightemp.json" \
    --input_path "/mnt/nfs/users/zwcolin/pillar_vlm_parent_vbc/lib/LLaMA-Factory/outputs/eval/32b_stage2_hightemp/generated_predictions.jsonl"


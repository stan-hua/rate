#### TODO: change the paths
gt_dir=/mnt/nfs/users/zwcolin/pillar_vlm_parent_vbc/lib/LLaMA-Factory/outputs/eval/rate_outputs/gt
pred_dir=/mnt/nfs/users/zwcolin/pillar_vlm_parent_vbc/lib/LLaMA-Factory/outputs/eval/rate_outputs/32b_stage2_hightemp_predictions

python -m src.evaluate \
    --gt $gt_dir/questions.csv \
    --pred $pred_dir/questions.csv \
    --save-path $pred_dir/evaluation.json \

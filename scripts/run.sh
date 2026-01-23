#### TODO: change the init environment
source /mnt/nfs/users/zwcolin/pillar_vlm_parent_vbc/lib/LLaMA-Factory/shell_scripts/local_env.sh

stages=("remove-comparisons" "extract-findings" "map-categories")

#### TODO: change the paths
filename=gt
input_dir=/mnt/nfs/users/zwcolin/pillar_vlm_parent_vbc/lib/LLaMA-Factory/outputs/eval/$filename.json
output_dir=/mnt/nfs/users/zwcolin/pillar_vlm_parent_vbc/lib/LLaMA-Factory/outputs/eval/rate_outputs/$filename

for stage in "${stages[@]}"; do
    python -m src.cli \
        --input-files $input_dir \
        --modality-config config/modalities/abdomen_ct.yaml \
        --save-dir $output_dir \
        --stages $stage 
done

python -m src.cli \
    --input-files $input_dir \
    --modality-config config/modalities/abdomen_ct.yaml \
    --save-dir $output_dir \
    --stages process-questions \
    --config config/process_questions.yaml # process-questions stage should be the last stage

source .venv/bin/activate
stages=("extract-findings" "map-categories")

filename=gt_32b_4ep
output_dir=/data/users/zwcolin/rate_outputs/$filename

for stage in "${stages[@]}"; do
    python -m src.cli \
        --input-files /data/users/zwcolin/$filename.json \
        --modality-config config/modalities/abdomen_ct.yaml \
        --save-dir $output_dir \
        --stages $stage 
done
python -m src.cli \
    --input-files /data/users/zwcolin/$filename.json \
    --modality-config config/modalities/abdomen_ct.yaml \
    --save-dir $output_dir \
    --stages process-questions \
    --config config/process_questions.yaml # process-questions stage should be the last stage

stages=("remove-comparisons" "extract-findings" "map-categories")
filename=pred_32b_4ep
output_dir=/data/users/zwcolin/rate_outputs/$filename

for stage in "${stages[@]}"; do
    python -m src.cli \
        --input-files /data/users/zwcolin/$filename.json \
        --modality-config config/modalities/abdomen_ct.yaml \
        --save-dir $output_dir \
        --stages $stage 
done
python -m src.cli \
    --input-files /data/users/zwcolin/$filename.json \
    --modality-config config/modalities/abdomen_ct.yaml \
    --save-dir $output_dir \
    --stages process-questions \
    --config config/process_questions.yaml # process-questions stage should be the last stage



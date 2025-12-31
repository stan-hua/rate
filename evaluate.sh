source .venv/bin/activate

python -m src.evaluate \
    --gt /data/users/zwcolin/rate_outputs/gt_32b_4ep/questions.csv \
    --pred /data/users/zwcolin/rate_outputs/pred_32b_4ep/questions.csv \
    --save-path /data/users/zwcolin/rate_outputs/pred_32b_4ep/evaluation.json \

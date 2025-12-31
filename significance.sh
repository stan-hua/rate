source .venv/bin/activate

python -m src.evaluate_bootstrap \
  --in-json /data/users/zwcolin/rate_outputs/pred_32b_4ep/evaluation.json \
  --out-json /data/users/zwcolin/rate_outputs/pred_32b_4ep/evaluation_plus_ci.json \
  --boot-reports 4000 \
  --subsample-sizes 50,100,200,500,1000,2000,5000

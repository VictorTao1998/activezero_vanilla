export PYTHONWARNINGS="ignore"

python /jianyu-fast-vol/activezero_vanilla/test_psmnet_with_confidence.py   \
--config-file 'configs/remote_test.yaml' \
--model '/jianyu-fast-vol/eval/sim_cv_bce_2/train_sim_cv_bce/models/model_100000.pth' \
--output '/jianyu-fast-vol/eval/sim_cv_bce_2/test_sim_cv_bce' \
--exclude-bg \
--exclude-zeros
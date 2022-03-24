export PYTHONWARNINGS="ignore"

python -m torch.distributed.launch --nproc_per_node=2 /jianyu-fast-vol/activezero_vanilla/train_psmnet_ir_reproj_p1.py   \
--config-file '/jianyu-fast-vol/activezero_vanilla/configs/remote_train_primitive_randscenes.yaml' \
--logdir '/jianyu-fast-vol/eval/final_exp/p1_wo_simgt' \
--gaussian-blur \
--color-jitter \
export PYTHONWARNINGS="ignore"

python -m torch.distributed.launch --nproc_per_node=2 /jianyu-fast-vol/activezero_vanilla/reproj_without_sim_gt/train_psmnet_ir_reproj_p2.py   \
--config-file '/jianyu-fast-vol/activezero_vanilla/configs/remote_train_primitive_randscenes.yaml' \
--logdir '/jianyu-fast-vol/eval/final_exp/p2_wo_simgt' \
--gaussian-blur \
--color-jitter \
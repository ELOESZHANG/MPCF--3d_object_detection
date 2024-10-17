#!/usr/bin/env bash
#creat kitti_pkl and gt
#python -m pcdet.datasets.kitti.kitti_dataset_custom create_kitti_infos ../tools/cfgs/dataset_configs/kitti_dataset_custom.yaml

##Train
##############  MPCF: 1 GPU ######################
python train.py --gpu_id 0 --workers 0 --cfg_file cfgs/kitti_models/mpcf.yaml \
   --batch_size 1 --epochs 60 --max_ckpt_save_num 25   \
 --fix_random_seed #--save_to_file

##Train
##############  MPCF: 4 GPUs ######################
#python -m torch.distributed.launch --nnodes 1 --nproc_per_node=4 --master_port 25511 train.py --gpu_id 0,1,2,3 --launch 'pytorch' --workers 4 \
#   --batch_size 4 --cfg_file cfgs/kitti_models/mpcf.yaml  --tcp_port 61000 --num_cpu 30 \
#   --epochs 40 --max_ckpt_save_num 30  \
#   --fix_random_seed #--save_to_file


##test
##############  MPCF: 1 GPU ######################
python test.py --gpu_id 1 --workers 4 --cfg_file cfgs/kitti_models/mpcf_test.yaml --batch_size 1 \
--ckpt ../output/kitti_models/mpcf/default/ckpt/checkpoint_epoch_1.pth #--save_to_file #






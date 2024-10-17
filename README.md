# MPCF--3d_object_detection
We propose a multi-phase consolidated fusion (MPCF) framework, a multimodal network favoring uniform spatial distribution for 3D object detection.




This is the official implementation of [**MPCF**], built on [`SFD`](https://github.com/LittlePey/SFD) and [`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet)


### Installation
1.  Prepare for the running environment. 

    You can  follow the installation steps in [`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet). We use 1 RTX-3090 or 4 RTX-4090 GPUs to train our MPCF.

2. Prepare for the data.  
    
    The dataset is follow SFD. Anyway, you should have your dataset as follows:

    ```
    MPCF
    ├── data
    │   ├── kitti_pseudo
    │   │   │── ImageSets
    │   │   │── training
    │   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & depth_dense_twise & depth_pseudo_rgbseguv_twise
    │   │   │── testing
    │   │   │   ├──calib & velodyne & image_2 & depth_dense_twise & depth_pseudo_rgbseguv_twise
    │   │   │── gt_database
    │   │   │── gt_database_pseudo_seguv
    │   │   │── kitti_dbinfos_train_custom_seguv.pkl
    │   │   │── kitti_infos_test.pkl
    │   │   │── kitti_infos_train.pkl
    │   │   │── kitti_infos_trainval.pkl
    │   │   │── kitti_infos_val.pkl
    ├── pcdet
    ├── tools
    ```
    .

3. Setup.

    ```
    conda create -n MPCF_env python=3.8
    conda activate MPCF_env
    
    pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements.txt
    pip install spconv-cu116

    cd MPCF
    python setup.py develop
    
    cd pcdet/ops/iou3d/cuda_op
    python setup.py develop

    ```

### Getting Started

   You can find the training and testing commands in tools/GP_run.sh

1. Training.

    For single GPU
    ```
    cd tools
    python train.py --gpu_id 0 --workers 0 --cfg_file cfgs/kitti_models/mpcf.yaml \
     --batch_size 1 --epochs 60 --max_ckpt_save_num 25 --fix_random_seed
    ```
    
    For 4 GPUs
    ```
    cd tools
    python -m torch.distributed.launch --nnodes 1 --nproc_per_node=4 --master_port 25511 train.py \
     --gpu_id 0,1,2,3 --launch 'pytorch' --workers 4 \
     --batch_size 4 --cfg_file cfgs/kitti_models/mpcf.yaml  --tcp_port 61000 \
     --epochs 40 --max_ckpt_save_num 30 --fix_random_seed
    ```

2. Evaluation.

    ```
    cd tools
    python test.py --gpu_id 1 --workers 4 --cfg_file cfgs/kitti_models/mpcf_test.yaml --batch_size 1 \
     --ckpt ../output/kitti_models/mpcf/default/ckpt/checkpoint_epoch_57.pth #--save_to_file 
    ```

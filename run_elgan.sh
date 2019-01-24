#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH -o elgan%j.out
python train.py --dataroot ../../../media/deepstorage01/datasets_external/facades --name facades_elgan --netD basic2 --model el --n_layers_D 5 --lr 0.0003 --lr_D 0.0001 --output_layer 4 --display_id 1 --D_train 200 --G_train 400 --pretrain_D 4000 --pretrain_G 6000 --model el --netG unet_256 --direction BtoA --lambda_EL 3 --dataset_mode aligned --norm batch --pool_size 0 --batch_size 8


#--dataroot 
#!/bin/sh
#SBATCH --job-name=train_tuiles
#SBATCH --output=slurm_out/tt.out
#SBATCH --error=slurm_out/tt.err
#SBATCH -p gpu-cbio
#SBATCH --gres=gpu:1 
#SBATCH --mem=40000
#SBATCH -c 10

module load cuda10.1

python train.py \
    --datadir /mnt/data4/jpaillard/projets/moco/HDF5/data/pcamv1/ \
    --epochs 50 \
    --batch_size 256 \
    --model_name perso \
    --name first_resnet \
    --weights_file /mnt/data4/jpaillard/projets/outputs/Hflip-Vflip-GrayScale-GaussianBlur-Crop-Rotate90-HEaug-200/checkpoint_0199.pth.tar \




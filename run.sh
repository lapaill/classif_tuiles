#!/bin/sh
#SBATCH --job-name=PCam-A-Geo
#SBATCH --output=slurm_out/A-Geo.out
#SBATCH --error=slurm_out/A-Geo.err
#SBATCH -p gpu-cbio
#SBATCH --gres=gpu:1
#SBATCH --mem=40000
#SBATCH -c 10

module load cuda10.1

python train.py \
    --datadir /mnt/data4/jpaillard/data/pcamv1 \
    --epochs 50 \
    --model_name perso \
    --name A-Geo \
    --batch_size 512 \
    --frozen \
    --pretrained \
    --weights_file /mnt/data4/jlaval/moco/outputs/GaussianBlur-GrayScale-Jitter-SameCrop-200/checkpoint_0143.pth.tar \


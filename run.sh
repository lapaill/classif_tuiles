#!/bin/sh
#SBATCH --job-name=PCam_baseline
#SBATCH --output=slurm_out/baseline.out
#SBATCH --error=slurm_out/baseline.err
#SBATCH -p gpu-cbio
#SBATCH --gres=gpu:1 
#SBATCH --mem=40000
#SBATCH -c 10

module load cuda10.1

python train.py \
    --datadir /mnt/data4/jpaillard/data/pcamv1 \
    --epochs 50 \
    --model_name resnet18 \
    --frozen \
    --pretrained \
    --name baseline \
    --batch_size 512 \
#    --weights_file /mnt/data4/jpaillard/projets/outputs/Hflip-Vflip-GrayScale-GaussianBlur-Crop-Rotate90-Jitter-200/checkpoint_0199.pth.tar \
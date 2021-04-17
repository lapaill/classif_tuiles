#!/bin/sh
#SBATCH --job-name=train_tuiles_jitter
#SBATCH --output=slurm_out/tt_jitter.out
#SBATCH --error=slurm_out/tt_jitter.err
#SBATCH -p gpu-cbio
#SBATCH --gres=gpu:1 
#SBATCH --mem=40000
#SBATCH -c 10

module load cuda10.1

python train.py \
    --datadir /mnt/data4/jpaillard/projets/moco/HDF5/data/pcamv1/ \
    --epochs 50 \
    --model_name perso \
    --frozen \
    --augmented \
    --pretrained \
    --name jitter \
    --batch_size 512 \
    --weights_file /mnt/data4/jpaillard/projets/outputs/Hflip-Vflip-GrayScale-GaussianBlur-Crop-Rotate90-Jitter-200/checkpoint_0199.pth.tar \
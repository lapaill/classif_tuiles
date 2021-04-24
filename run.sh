#!/bin/sh
#SBATCH --job-name=PCam_Elastic
#SBATCH --output=slurm_out/Elastic.out
#SBATCH --error=slurm_out/Elastic.err
#SBATCH -p gpu-cbio
#SBATCH --gres=gpu:1
#SBATCH --mem=40000
#SBATCH -c 10

module load cuda10.1

python train.py \
    --datadir /mnt/data4/jpaillard/data/pcamv1 \
    --epochs 50 \
    --model_name resnet18 \
    --name Elastic \
    --batch_size 512 \
    --frozen \
    --pretrained \
    --weights_file /mnt/data4/jlaval/moco/outputs/Hflip-Vflip-GrayScale-GaussianBlur-Rotate90-Jitter-MultipleElasticDistort-resume200/checkpoint_0199.pth.tar \


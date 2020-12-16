#!/bin/sh
#SBATCH --job-name=train_tuiles
#SBATCH --output=slurm_out/tt.out
#SBATCH --error=slurm_out/tt.err
#SBATCH -p gpu-cbio
#SBATCH --gres=gpu:1 
#SBATCH --mem=40000

module load cuda10.1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate histo


python train.py --datadir data/test_0 --pretrained --model_name resnet50 --epochs 100 --batch_size 64



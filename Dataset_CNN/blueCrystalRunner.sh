#!/bin/bash -login
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --job-name=dataScience
#SBATCH --mem=100G
#SBATCH -t 72:00:00
module load CUDA
 
####--cpus-per-task=28

source /mnt/storage/home/zx18522/miniconda3/bin/activate DataScience

python hello.py 

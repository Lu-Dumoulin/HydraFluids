#!/bin/env bash
#SBATCH --array=1-12:4
#SBATCH --partition=private-kruse-gpu,shared-gpu
#SBATCH --time=0-12:00:00
#SBATCH --output=%J.out
#SBATCH --mem=3000  
#SBATCH --gpus=1 
#SBATCH --constraint=nvidia_h100_nvl|nvidia_a100-pcie-40gb

export use_gpu=true
export path_to_data=/srv/beegfs/scratch/users/d/dumoulil/Data/PQ_FFT/

mkdir -p $path_to_data

module load Julia

cd $path_to_data
srun julia --optimize=3 /home/users/d/dumoulil/Code/PQ_FFT/main.jl
		
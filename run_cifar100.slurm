#!/bin/bash --login
#SBATCH -J cifar100
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 1-00:00:00
#SBATCH -o cifar100.%j.out
#SBATCH -e cifar100.%j.err

set -euo pipefail

module purge
module load cuda/12.6.2

source "$HOME/scratch/cifar100-venv/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR="$PWD/outputs/.matplotlib"

mkdir -p outputs

echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $PWD"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
python --version

python src/cifar100_experiments.py \
  --preset coursework \
  --epochs 50 \
  --device cuda \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}"

echo "Finished at: $(date)"

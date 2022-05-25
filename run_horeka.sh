#!/usr/bin/env bash
# `bash -x` for detailed Shell debugging

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --gres gpu:4
#SBATCH --partition accelerated
#SBATCH --exclusive

mkdir /tmp/data/
#module load UnZip

unzip /hkfs/home/dataset/datasets/So2Sat_POP/So2Sat_POP_Part1.zip -d /tmp/data/
unzip /hkfs/home/dataset/datasets/So2Sat_POP/So2Sat_POP_Part2.zip -d /tmp/data/

cd /home/haicore-project-fzj/fzj_al.bazarova/countmein_submission

ml --force  purge

#srun python starter-pack3.py 
srun singularity run --bind /home/haicore-project-fzj/fzj_al.bazarova/countmein_submission countmein_sklearn_1.0.sif  python3 starter-pack3.py


#!/usr/bin/env bash
# `bash -x` for detailed Shell debugging

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=8:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --account=hai_countmein

ml --force  purge

module load  Stages/2022  GCCcore/.11.2.0 Singularity-Tools/2022

#sleep 7000

mkdir /tmp/data/

module load UnZip

unzip /p/project/hai_countmein/data/So2Sat_POP_Part1.zip -d /tmp/data/
unzip /p/project/hai_countmein/data/So2Sat_POP_Part2.zip -d /tmp/data/

cd /home/haicore-project-fzj/fzj_al.bazarova/countmein_submission

# srun python train.py 
srun singularity run /p/project/hai_countmein/countmein_sklearn_1.0.sif  python3 starter-pack3.py
#--data_path_So2Sat_pop_part1 '/p/project/hai_countmein/data/So2Sat_POP_Part1' --data_path_So2Sat_pop_part2 '/p/project/hai_countmein/data/So2Sat_POP_Part2'


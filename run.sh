#!/usr/bin/env bash
# `bash -x` for detailed Shell debugging

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
##SBATCH --output=/p/home/jusers/benassou1/juwels/benassou1/hai_countmein/benassou1/Countmein_oursol/slurm-rf-%j.out
##SBATCH --error=/p/home/jusers/benassou1/juwels/benassou1/hai_countmein/benassou1/Countmein_oursol/slurm-rf-%j.err
#SBATCH --time=5:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --account=hai_countmein

# ml purge
ml GCC
ml PyTorch
ml GDAL
ml matplotlib
ml scikit-learn
# ml Stages/2020  GCC/9.3.0  ParaStationMPI/5.4.7-1 OpenCV/4.5.0-Python-3.8.5
# ml GDAL/3.1.2-Python-3.8.5
# ml scikit 

#cd /p/home/jusers/benassou1/juwels/bazarova1/hai_countmein/Countmein_oursol
#source sc_venv_template/venv/bin/activate
source /p/home/jusers/bazarova1/juwels/hai_countmein/Countmein_oursol/sc_venv_template/activate.sh

mkdir /tmp/data/

module load UnZip

unzip /p/project/hai_countmein/data/So2Sat_POP_Part1.zip -d /tmp/data/
unzip /p/project/hai_countmein/data/So2Sat_POP_Part2.zip -d /tmp/data/
#tar xvf $HOME/hai_countmein/So2Sat-POP/countmein1.tar -C /tmp/data/
#python3 --version
#which python3
#module unload Python/3.8.6
#which python3
#srun --ntasks-per-node=1  python3 -m  a.py
#python3 a.py

#srun python3 train.py -i /p/project/atmlaml/bazarova1/data/inputs -o /p/project/atmlaml/bazarova1/out -t TAF1
#srun python3 train.py -i /tmp/data/inputs -o /p/project/atmlaml/bazarova1/out_att_single-20-04-22 -t TAF1 -s 2048 -g -u -ce -a -r
#srun python predict.py -c liver -i $PROJECT_atmlaml/bazarova1/data/inputs -m $PROJECT_atmlaml/bazarova1/out_att_qkv_pe-20-04-22/best_model.h5 -o out_att_qkv_pe-20-04-22.gzip -p $PROJECT_atmlaml/bazarova1/data/inputs/label/predict_region.bed -b 2048 -ce -u

#cd starter-pack

#srun python starter-pack.py --data_path_So2Sat_pop_part1 'So2Sat POP Part1' --data_path_So2Sat_pop_part2 'So2Sat_POP_Part2'


#cd So2Sat-POP
cd /p/project/hai_countmein/countmein_submission
#cd starter-pack
#tar xvf csvs.tar -C /tmp/data/

# srun python train.py 
srun python starter-pack2.py
#srun python standard_sol.py #--data_path_So2Sat_pop_part1 '/p/project/hai_countmein/data/So2Sat_POP_Part1' --data_path_So2Sat_pop_part2 '/p/project/hai_countmein/data/So2Sat_POP_Part2'

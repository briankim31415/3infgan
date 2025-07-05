#!/bin/bash

#SBATCH -J inf_gan_geolife_simple_basic
#SBATCH -o run_logs/geolife_simple_basic/job.o%j
#SBATCH -e run_logs/geolife_simple_basic/job.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=briankim31415@gmail.com
#SBATCH -A CCR25007

cd ..
source ./venv/bin/activate
module load gcc cuda python3
python3 -m src.run --use_wandb --online --cfg_name=geolife_simple_basic
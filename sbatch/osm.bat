#!/bin/bash

#SBATCH -J inf_gan_osm
#SBATCH -o run_logs/osm/job.o%j
#SBATCH -e run_logs/osm/job.e%j
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=briankim31415@gmail.com
#SBATCH -A ASC24027

cd ..
source ./venv/bin/activate
module load gcc cuda python3
python3 -m src.run --use_wandb --online --cfg_name=osm
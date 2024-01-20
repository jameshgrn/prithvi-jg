#!/bin/bash

#SBATCH --mail-user=jhgearon@iu.edu
#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH -A r00268
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=0-2:59:00
#SBATCH --mem=128gb
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --job-name=levity
#SBATCH --output=lev.out
#SBATCH --error=lev.err

module load python/gpu/3.10.10

source /N/u/jhgearon/Quartz/.bashrc

pip install -r requirements.txt

img_path = /N/u/jhgearon/Quartz/prithvi_jg/images
t1.tif = img_path + t1.tif
t2.tif = img_path + t2.tif
t3.tif = img_path + t3.tif

cd /N/u/jhgearon/Quartz/Prithvi_100M

python Prithvi_run_inference.py --data_files t1.tif t2.tif t3.tif --yaml_file_path /path/to/yaml/Prithvi_100.yaml --checkpoint /Prithvi_100.pth --output_dir /path/to/out/dir/ --mask_ratio 0.5

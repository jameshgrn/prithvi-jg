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

cd /N/u/jhgearon/Quartz/prithvi-jg

pip install -r requirements.txt

img_path="/N/u/jhgearon/Quartz/prithvi-jg/images/"
t1="${img_path}t1.tif"
t2="${img_path}t2.tif"
t3="${img_path}t3.tif"

python Prithvi_run_inference.py --data_files $t1 $t2 $t3 --yaml_file_path Prithvi_100M_config.yaml --checkpoint Prithvi_100.pt --output_dir output --mask_ratio 0.5

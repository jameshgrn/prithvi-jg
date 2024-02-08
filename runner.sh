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
#SBATCH --job-name=prithvi
#SBATCH --output=prithvi.out
#SBATCH --error=prithvi.err

module load python/gpu/3.10.10

cd /N/u/jhgearon/Quartz/prithvi-jg

pip install -r requirements.txt

img_path="/N/u/jhgearon/Quartz/prithvi-jg/images/"
t1="${img_path}t1.tif"
t2="${img_path}t2.tif"
t3="${img_path}t3.tif"

# Check if images exist
if [[ ! -f "$t1" || ! -f "$t2" || ! -f "$t3" ]]; then
    python image_retrieval.py
fi

python Prithvi_run_inference.py --data_files $t1 $t2 $t3 --yaml_file_path Prithvi_100M_config.yaml --checkpoint Prithvi_100M.pt --output_dir output --mask_ratio 0.5
python image_plotting.py
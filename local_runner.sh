img_path="images/"
t1="${img_path}t1.tif"
t2="${img_path}t2.tif"
t3="${img_path}t3.tif"

# Check if images exist
if [[ ! -f "$t1" || ! -f "$t2" || ! -f "$t3" ]]; then
   poetry run python image_retrieval.py
fi

poetry run python Prithvi_run_inference.py --data_files $t1 $t2 $t3 --yaml_file_path Prithvi_100M_config.yaml --checkpoint Prithvi_100M.pt --output_dir output --mask_ratio 0.5
poetry run python image_plotting.py


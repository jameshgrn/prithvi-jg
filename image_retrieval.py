# Import necessary libraries
from pystac_client import Client
from odc.stac import load
import rioxarray
import xarray as xr
import numpy as np
import geopandas as gpd
import yaml
import os
import cv2
import requests
from io import BytesIO
from Prithvi_run_inference import _convert_np_uint8

# Import custom functions from image_analysis module
from image_formatting import load_example, predict_on_images, save_geotiff

# Initialize STAC client and define constants
client = Client.open("https://earth-search.aws.element84.com/v1")
collection = "sentinel-2-l2a"
tas_bbox = [-72.33199899636085, 8.597982612737141, -72.28956118968598, 8.655018875697806]
time_intervals = ["2019-01-01/2019-12-31", "2020-01-01/2020-12-31", "2023-01-01/2023-12-31"]
output_files = ["images/t1.tif", "images/t2.tif", "images/t3.tif"]
scale_factor = 0.0001  # Scale factor for converting raw data numbers into reflectance

# Function to download an image from a URL
def download_image(url):
    """Download an image from a given URL."""
    response = requests.get(url)
    if response.status_code == 200:
        return cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        print(f"Error downloading {url}, status code: {response.status_code}")
        return None

# Function to resize images to a common shape and save them
def resize_images_to_common_shape_and_save(image_file_paths, output_dir="resized_images"):
    """Resize images to a common shape and save them to a specified directory."""
    target_shape = (256, 256)  # Example target shape
    resized_image_paths = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for url in image_file_paths:
        img = download_image(url)
        if img is not None:
            resized_img = cv2.resize(img, target_shape, interpolation=cv2.INTER_AREA)
            base_name = os.path.basename(url)
            new_path = os.path.join(output_dir, base_name)
            cv2.imwrite(new_path, resized_img)
            resized_image_paths.append(new_path)
        else:
            print(f"Failed to process image from {url}")
    return resized_image_paths

# Main process
if __name__ == "__main__":
    # Check for the existence of required files
    existing_files = [f for f in output_files if os.path.isfile(f)]
    if len(existing_files) == len(output_files):
        print("All required tif files exist. Skipping download.")
    else:
        # Load configuration
        with open('Prithvi_100M_config.yaml') as f:
            config = yaml.safe_load(f)

        for i, time_interval in enumerate(time_intervals):
            if output_files[i] in existing_files:
                print(f"File {output_files[i]} already exists. Skipping download for this time interval.")
                continue

            # Search and process images
            search = client.search(collections=[collection], bbox=tas_bbox, datetime=time_interval, limit=100)
            items = list(search.item_collection())
            if not items:
                print(f"No images found for time interval {time_interval}")
                continue

            selected_item = min(items, key=lambda item: item.properties["eo:cloud_cover"])
            print(f"Selected item {selected_item.id} with cloud cover: {selected_item.properties['eo:cloud_cover']}%")
            
            bands_to_save = ['blue', 'red', 'green', 'nir', 'swir16', 'swir22']
            image_file_paths = [selected_item.assets[band].href for band in bands_to_save]
            resized_image_paths = resize_images_to_common_shape_and_save(image_file_paths)
            
            # Image analysis and saving results
            outputs = predict_on_images(data_files=resized_image_paths, mask_ratio=0.5, yaml_file_path='Prithvi_100M_config.yaml', checkpoint='checkpoints/Prithvi_100M.pth')
            for t, (input_img, rec_img, mask_img) in enumerate(zip(outputs[0], outputs[1], outputs[2])):
                save_geotiff(image=_convert_np_uint8(input_img), output_path=f"original_rgb_t{t}.tiff", meta=meta_data[t])
                save_geotiff(image=_convert_np_uint8(rec_img), output_path=f"predicted_rgb_t{t}.tiff", meta=meta_data[t])
                save_geotiff(image=_convert_np_uint8(mask_img), output_path=f"masked_rgb_t{t}.tiff", meta=meta_data[t])

        # Update configuration based on analysis results
        with open('Prithvi_100M_config.yaml', 'w') as f:
            yaml.safe_dump(config, f)

        # Note: The function _convert_np_uint8 is assumed to be defined elsewhere in the script or in an imported module.
        # It should convert numpy arrays to uint8 format, suitable for image saving.

# Additional utility functions or classes can be defined here if necessary
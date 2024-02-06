from pystac_client import Client
from odc.stac import load
import rioxarray
import xarray as xr
import numpy as np
import geopandas as gpd
import yaml
import os
import cv2
import os
import numpy as np
import requests
from io import BytesIO

# Add necessary imports from image_analysis
from image_formatting import load_example, predict_on_images, save_geotiff

client = Client.open("https://earth-search.aws.element84.com/v1")
collection = "sentinel-2-l2a"

# new bbox
tas_bbox = [-72.33199899636085, 8.597982612737141, -72.28956118968598, 8.655018875697806]
# Define the time intervals
time_intervals = ["2019-01-01/2019-12-31", "2020-01-01/2020-12-31", "2023-01-01/2023-12-31"]
output_files = ["images/t1.tif", "images/t2.tif", "images/t3.tif"]
# Scale factor for converting raw data numbers into reflectance
scale_factor = 0.0001


def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_UNCHANGED)
        return image
    else:
        print(f"Error downloading {url}, status code: {response.status_code}")
        return None

def resize_images_to_common_shape_and_save(image_file_paths, output_dir="resized_images"):
    target_shape = (256, 256)  # Example target shape
    resized_image_paths = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for url in image_file_paths:
        img = download_image(url)
        if img is not None:
            resized_img = cv2.resize(img, target_shape, interpolation=cv2.INTER_AREA)
            # Generate a new file path
            base_name = os.path.basename(url)
            new_path = os.path.join(output_dir, base_name)
            cv2.imwrite(new_path, resized_img)
            resized_image_paths.append(new_path)
        else:
            print(f"Failed to process image from {url}")
    return resized_image_paths

# Check if the data t1.tif, t2.tif, and t3.tif exist in the images folder
existing_files = [f for f in output_files if os.path.isfile(f)]

# If files exist, skip the download and proceed to formatting
if len(existing_files) == len(output_files):
    print("All required tif files exist. Skipping download.")
else:
    # Define a function to calculate mean and std for each band
    def calculate_band_statistics(xr_dataset, bands):
        stats = {}
        for band in bands:
            stats[band] = {
                'mean': float(xr_dataset[band].mean().values),
                'std': float(xr_dataset[band].std().values)
            }
        return stats

    # Load the YAML configuration file before the loop if the config is not meant to change per iteration
    with open('Prithvi_100M_config.yaml') as f:
        config = yaml.safe_load(f)

    for i, time_interval in enumerate(time_intervals):
        if output_files[i] in existing_files:
            print(f"File {output_files[i]} already exists. Skipping download for this time interval.")
            continue

        # Search for items in the collection within the time interval
        search = client.search(
            collections=[collection],
            bbox=tas_bbox,
            datetime=time_interval,
            limit=100)  # Increase limit if needed)
        items = list(search.item_collection())
        
        if not items:
            print(f"No images found for time interval {time_interval}")
            continue

        # Sort the items by cloud cover and select the one with the lowest value
        selected_item = min(items, key=lambda item: item.properties["eo:cloud_cover"])
        print(f"Selected item {selected_item.id} with cloud cover: {selected_item.properties['eo:cloud_cover']}%")
        
        bands_to_save = ['blue', 'red', 'green', 'nir', 'swir16', 'swir22']
        mask_ratio = 0.5
        checkpoint_path = 'checkpoints/Prithvi_100M.pth'

        # Assuming bands_to_save is a list of band names that correspond to keys in the assets dictionary
        image_file_paths = [selected_item.assets[band].href for band in bands_to_save]

        # Load the images using the method from image_analysis
        resized_image_paths = resize_images_to_common_shape_and_save(image_file_paths)
        image_data, meta_data = load_example(file_paths=resized_image_paths, mean=config['train_params']['data_mean'], std=config['train_params']['data_std'])
        # Assuming image_file_paths contains URLs to all the images
        # Select only the first three images
        selected_image_paths = image_file_paths[:3]
        # Process and analyze the images
        outputs = predict_on_images(data_files=selected_image_paths, mask_ratio=mask_ratio, yaml_file_path='Prithvi_100M_config.yaml', checkpoint=checkpoint_path)

        # Save the results as GeoTIFF
        for t, (input_img, rec_img, mask_img) in enumerate(zip(outputs[0], outputs[1], outputs[2])):
            save_geotiff(image=_convert_np_uint8(input_img), output_path=f"original_rgb_t{t}.tiff", meta=meta_data[t])
            save_geotiff(image=_convert_np_uint8(rec_img), output_path=f"predicted_rgb_t{t}.tiff", meta=meta_data[t])
            save_geotiff(image=_convert_np_uint8(mask_img), output_path=f"masked_rgb_t{t}.tiff", meta=meta_data[t])

        # Now update the YAML configuration file
        with open('Prithvi_100M_config.yaml') as f:
            config = yaml.safe_load(f)

        # Update the means and stds in the config
        config['train_params']['data_mean'] = [band_statistics[band]['mean'] for band in bands_to_save]
        config['train_params']['data_std'] = [band_statistics[band]['std'] for band in bands_to_save]

        # Write the updated configuration back to the file
        with open('Prithvi_100M_config.yaml', 'w') as f:
            yaml.safe_dump(config, f)

        # Write the dataset to a raster file
        xr_dataset.rio.to_raster(output_files[i])


from pystac_client import Client
from odc.stac import load
import rioxarray
import xarray as xr
import numpy as np
import geopandas as gpd
import yaml

# Add necessary imports from image_analysis
from image_analysis import load_example, predict_on_images, save_geotiff

client = Client.open("https://earth-search.aws.element84.com/v1")
collection = "sentinel-2-l2a"

# new bbox
tas_bbox = [-72.33199899636085, 8.597982612737141, -72.28956118968598, 8.655018875697806]
# Define the time intervals
time_intervals = ["2019-01-01/2019-12-31", "2020-01-01/2020-12-31", "2023-01-01/2023-12-31"]
output_files = ["images/t1.tif", "images/t2.tif", "images/t3.tif"]
# Scale factor for converting raw data numbers into reflectance
scale_factor = 0.0001
# Define the maximum acceptable cloud cover percentage

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
    image_data, meta_data = load_example(file_paths=image_file_paths, mean=config['train_params']['data_mean'], std=config['train_params']['data_std'])

    # Process and analyze the images
    outputs = predict_on_images(data_files=image_file_paths, mask_ratio=mask_ratio, yaml_file_path='Prithvi_100M_config.yaml', checkpoint=checkpoint_path)

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


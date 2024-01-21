from pystac_client import Client
from odc.stac import load
import rioxarray
import xarray as xr
import numpy as np
import geopandas as gpd
import yaml

client = Client.open("https://earth-search.aws.element84.com/v1")
collection = "sentinel-2-l2a"

# new bbox
tas_bbox = [146.5, -43.6, 146.7, -43.4]
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

for i, time_interval in enumerate(time_intervals):
    # Search for items in the collection within the time interval
    search = client.search(
        collections=[collection],
        bbox=tas_bbox,
        datetime=time_interval,
        limit=100)  # Increase limit if needed)
    items = list(search.get_all_items())
    
    if not items:
        print(f"No images found for time interval {time_interval}")
        continue

    # Sort the items by cloud cover and select the one with the lowest value
    selected_item = min(items, key=lambda item: item.properties["eo:cloud_cover"])
    print(f"Selected item {selected_item.id} with cloud cover: {selected_item.properties['eo:cloud_cover']}%")
    
    # Load the data using odc-stac
    data = load([selected_item], bbox=tas_bbox, groupby="solar_day", chunks={}, crs="EPSG:3857")
    
    # Create a Dataset with each band as a separate variable
    xr_dataset = xr.Dataset()
    bands_to_save = ['blue', 'green', 'red', 'nir', 'swir16', 'swir22']
    for band_name in bands_to_save:
        # Assuming data[band_name] is a DataArray with the band data
        xr_dataset[band_name] = data[band_name].isel(time=0) #* scale_factor

    # Set the CRS for the dataset
    xr_dataset.rio.write_crs(data['spatial_ref'].attrs['crs_wkt'], inplace=True)

    # After creating the xr_dataset with all bands
    band_statistics = calculate_band_statistics(xr_dataset, bands_to_save)

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


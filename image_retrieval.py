# Import necessary libraries
from pystac_client import Client
import os
import requests
import rasterio
from rasterio.merge import merge
import numpy as np
from odc.stac import load
import odc.geo
import rioxarray

# Initialize STAC client and define constants
client = Client.open("https://earth-search.aws.element84.com/v1")
collection = "sentinel-2-l2a"
tas_bbox = [-72.33199899636085, 8.597982612737141, -72.28956118968598, 8.655018875697806]
time_intervals = ["2019-01-01/2019-12-31", "2020-01-01/2020-12-31", "2023-01-01/2023-12-31"]
output_dir = "images"
bands_order = ['blue', 'red', 'green', 'nir', 'swir16', 'swir22']

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def stack_bands(band_paths, output_path):
    """Stack bands in the specified order and save as a single multi-band TIFF."""
    bands_data = []
    for path in band_paths:
        with rasterio.open(path) as src:
            bands_data.append(src.read(1))
    # Stack arrays along new dimension
    multi_band_array = np.stack(bands_data, axis=0)
    # Use the metadata of the first band as a template
    with rasterio.open(band_paths[0]) as src:
        meta = src.meta
    meta.update(count=len(band_paths))
    with rasterio.open(output_path, 'w', **meta) as dst:
        for i, layer in enumerate(multi_band_array, start=1):
            dst.write(layer, i)

# Main process
if __name__ == "__main__":
    for i, time_interval in enumerate(time_intervals):
        search = client.search(collections=[collection], bbox=tas_bbox, datetime=time_interval, max_items=10)
        bands_order = ['blue', 'red', 'green', 'nir', 'swir16', 'swir22']
        items = list(search.item_collection())
        # selected_item = min(items, key=lambda item: item.properties["eo:cloud_cover"])
        # print(f"Selected item {selected_item.id} with cloud cover: {selected_item.properties['eo:cloud_cover']}%")
        data = load(search.items(), bbox=tas_bbox, groupby="solar_day", chunks={}, bands=bands_order)
        output_path = os.path.join(output_dir, f"t{i}.tif")
        data.isel(time=0).rio.to_raster(output_path)
        print(f"Saved {output_path}")
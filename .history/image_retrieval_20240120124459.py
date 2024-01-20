from pystac_client import Client
from odc.stac import load
import rasterio

client = Client.open("https://earth-search.aws.element84.com/v1")
collection = "sentinel-2-l2a"
tas_bbox = [146.5, -43.6, 146.7, -43.4]

# Define the time intervals
time_intervals = ["2023-01", "2023-06", "2023-12"]
output_files = ["/N/u/jhgearon/Quartz/prithvi_jg/images/t1.tif", "/N/u/jhgearon/Quartz/prithvi_jg/images/t2.tif", "/N/u/jhgearon/Quartz/prithvi_jg/images/t3.tif"]

for i, time_interval in enumerate(time_intervals):
    search = client.search(collections=[collection], bbox=tas_bbox, datetime=time_interval)
    data = load(search.items(), bbox=tas_bbox, groupby="solar_day", chunks={})
    # Save the red, green, blue bands of the image to a tif file
with rasterio.open(output_files[i], 'w', driver='GTiff', height=data['red'].shape[1], width=data['red'].shape[2], count=3, dtype=data['red'].dtype) as dst:
        dst.write(data[["red", "green", "blue"]].isel(time=0).to_array())

from pystac_client import Client
from odc.stac import load
import rasterio

client = Client.open("https://earth-search.aws.element84.com/v1")
collection = "sentinel-2-l2a"
tas_bbox = [146.5, -43.6, 146.7, -43.4]

# Define the time intervals
time_intervals = ["2023-01", "2023-06", "2023-12"]
#output_files = ["/N/u/jhgearon/Quartz/prithvi-jg/images/t1.tif", "/N/u/jhgearon/Quartz/prithvi-jg/images/t2.tif", "/N/u/jhgearon/Quartz/prithvi-jg/images/t3.tif"]
output_files = ["t1.tif", "t2.tif", "t3.tif"]
for i, time_interval in enumerate(time_intervals):
    search = client.search(collections=[collection], bbox=tas_bbox, datetime=time_interval)
    data = load(search.items(), bbox=tas_bbox, groupby="solar_day", chunks={})
    print(data.)
    # Assuming 'narrow_nir', 'swir1', and 'swir2' are the names of the bands in the dataset
    bands_to_save = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    band_count = len(bands_to_save)
    # Save the specified bands of the image to a tif file
    with rasterio.open(output_files[i], 'w', driver='GTiff', 
                       height=data['red'].shape[1], 
                       width=data['red'].shape[2], 
                       count=band_count, 
                       dtype=data['red'].dtype) as dst:
        for j, band in enumerate(bands_to_save):
            dst.write(data[band].isel(time=0).values, j + 1)

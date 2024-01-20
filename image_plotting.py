import matplotlib.pyplot as plt
import numpy as np
import rasterio

# Define file paths for the TIFFs, masks, and predictions
tiff_files = ["images/t1.tif", "images/t2.tif", "images/t3.tif"]
mask_files = ["output/mask_t0.tiff", "output/mask_t1.tiff", "output/mask_t2.tiff"]
prediction_files = ["output/predicted_t0.tiff", "output/predicted_t1.tiff", "output/predicted_t2.tiff"]

# Function to read and stack the RGB bands of a TIFF
def read_rgb_image(tiff_path):
    with rasterio.open(tiff_path) as src:
        red = src.read(3)
        green = src.read(2)
        blue = src.read(1)
    return np.stack((red, green, blue), axis=-1)

# Function to read a single-band image (mask or prediction)
def read_single_band_image(image_path):
    with rasterio.open(image_path) as src:
        return src.read(1)

# Create a 3x3 plot matrix
fig, axs = plt.subplots(3, 3, figsize=(5, 8))

for i in range(3):
    # Plot original TIFFs (RGB)
    rgb_image = read_rgb_image(tiff_files[i])
    axs[0, i].imshow(rgb_image)
    axs[0, i].axis('off')
    axs[0, i].set_title(f'Original TIFF {i+1}')

    # Plot masks
    mask_image = read_single_band_image(mask_files[i])
    axs[1, i].imshow(mask_image, cmap='gray')
    axs[1, i].axis('off')
    axs[1, i].set_title(f'Mask {i+1}')

    # Plot predictions
    prediction_image = read_single_band_image(prediction_files[i])
    axs[2, i].imshow(prediction_image, cmap='gray')
    axs[2, i].axis('off')
    axs[2, i].set_title(f'Prediction {i+1}')

# Adjust layout
plt.tight_layout()
plt.savefig('output/plot.png', dpi=75)
plt.show()


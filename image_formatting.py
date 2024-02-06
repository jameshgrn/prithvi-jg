import argparse
import functools
import os
from typing import List

import numpy as np
import rasterio
import torch
import yaml
from einops import rearrange

from Prithvi import MaskedAutoencoderViT
import gradio as gr
from functools import partial


NO_DATA = -9999
NO_DATA_FLOAT = 0.0001
PERCENTILES = (0.1, 99.9)


def process_channel_group(orig_img, new_img, channels, data_mean, data_std):
    """ Process *orig_img* and *new_img* for RGB visualization. Each band is rescaled back to the
        original range using *data_mean* and *data_std* and then lowest and highest percentiles are
        removed to enhance contrast. Data is rescaled to (0, 1) range and stacked channels_first.
    Args:
        orig_img: torch.Tensor representing original image (reference) with shape = (bands, H, W).
        new_img: torch.Tensor representing image with shape = (bands, H, W).
        channels: list of indices representing RGB channels.
        data_mean: list of mean values for each band.
        data_std: list of std values for each band.
    Returns:
        torch.Tensor with shape (num_channels, height, width) for original image
        torch.Tensor with shape (num_channels, height, width) for the other image
    """

    stack_c = [], []

    for c in channels:
        orig_ch = orig_img[c, ...]
        valid_mask = torch.ones_like(orig_ch, dtype=torch.bool)
        valid_mask[orig_ch == NO_DATA_FLOAT] = False

        # Back to original data range
        orig_ch = (orig_ch * data_std[c]) + data_mean[c]
        new_ch = (new_img[c, ...] * data_std[c]) + data_mean[c]

        # Rescale (enhancing contrast)
        min_value, max_value = np.percentile(orig_ch[valid_mask], PERCENTILES)

        orig_ch = torch.clamp((orig_ch - min_value) / (max_value - min_value), 0, 1)
        new_ch = torch.clamp((new_ch - min_value) / (max_value - min_value), 0, 1)

        # No data as zeros
        orig_ch[~valid_mask] = 0
        new_ch[~valid_mask] = 0

        stack_c[0].append(orig_ch)
        stack_c[1].append(new_ch)

    # Channels first
    stack_orig = torch.stack(stack_c[0], dim=0)
    stack_rec = torch.stack(stack_c[1], dim=0)

    return stack_orig, stack_rec


def read_geotiff(file_path: str):
    """ Read all bands from *file_path* and returns image + meta info.
    Args:
        file_path: path to image file.
    Returns:
        np.ndarray with shape (bands, height, width)
        meta info dict
    """

    with rasterio.open(file_path) as src:
        img = src.read()
        meta = src.meta

    return img, meta


def save_geotiff(image, output_path: str, meta: dict):
    """ Save multi-band image in Geotiff file.
    Args:
        image: np.ndarray with shape (bands, height, width)
        output_path: path where to save the image
        meta: dict with meta info.
    """

    with rasterio.open(output_path, "w", **meta) as dest:
        for i in range(image.shape[0]):
            dest.write(image[i, :, :], i + 1)

    return


def _convert_np_uint8(float_image: torch.Tensor):

    image = float_image.numpy() * 255.0
    image = image.astype(dtype=np.uint8)
    image = image.transpose((1, 2, 0))

    return image


def load_example(file_paths: List[str], mean: List[float], std: List[float]):
    """ Build an input example by loading images in *file_paths*.
    Args:
        file_paths: list of file paths .
        mean: list containing mean values for each band in the images in *file_paths*.
        std: list containing std values for each band in the images in *file_paths*.
    Returns:
        np.array containing created example
        list of meta info for each image in *file_paths*
    """

    imgs = []
    metas = []

    for file in file_paths:
        img, meta = read_geotiff(file)
        img = img[:6]*10000 if img[:6].mean() <= 2 else img[:6]

        # Rescaling (don't normalize on nodata)
        img = np.moveaxis(img, 0, -1)   # channels last for rescaling
        img = np.where(img == NO_DATA, NO_DATA_FLOAT, (img - mean) / std)

        imgs.append(img)
        metas.append(meta)

    imgs = np.stack(imgs, axis=0)    # num_frames, H, W, C
    imgs = np.moveaxis(imgs, -1, 0).astype('float32')  # C, num_frames, H, W
    imgs = np.expand_dims(imgs, axis=0)  # add batch dim

    return imgs, metas


def run_model(model: torch.nn.Module, input_data: torch.Tensor, mask_ratio: float, device: torch.device):
    """ Run *model* with *input_data* and create images from output tokens (mask, reconstructed + visible).
    Args:
        model: MAE model to run.
        input_data: torch.Tensor with shape (B, C, T, H, W).
        mask_ratio: mask ratio to use.
        device: device where model should run.
    Returns:
        3 torch.Tensor with shape (B, C, T, H, W).
    """

    with torch.no_grad():
        x = input_data.to(device)

        _, pred, mask = model(x, mask_ratio)

    # Create mask and prediction images (un-patchify)
    mask_img = model.unpatchify(mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1])).detach().cpu()
    pred_img = model.unpatchify(pred).detach().cpu()

    # Mix visible and predicted patches
    rec_img = input_data.clone()
    rec_img[mask_img == 1] = pred_img[mask_img == 1]  # binary mask: 0 is keep, 1 is remove

    # Switch zeros/ones in mask images so masked patches appear darker in plots (better visualization)
    mask_img = (~(mask_img.to(torch.bool))).to(torch.float)

    return rec_img, mask_img


def save_rgb_imgs(input_img, rec_img, mask_img, channels, mean, std, output_dir, meta_data):
    """ Wrapper function to save Geotiff images (original, reconstructed, masked) per timestamp.
    Args:
        input_img: input torch.Tensor with shape (C, T, H, W).
        rec_img: reconstructed torch.Tensor with shape (C, T, H, W).
        mask_img: mask torch.Tensor with shape (C, T, H, W).
        channels: list of indices representing RGB channels.
        mean: list of mean values for each band.
        std: list of std values for each band.
        output_dir: directory where to save outputs.
        meta_data: list of dicts with geotiff meta info.
    """

    for t in range(input_img.shape[1]):
        rgb_orig, rgb_pred = process_channel_group(orig_img=input_img[:, t, :, :],
                                                   new_img=rec_img[:, t, :, :],
                                                   channels=channels, data_mean=mean,
                                                   data_std=std)

        rgb_mask = mask_img[channels, t, :, :] * rgb_orig

        # Saving images

        save_geotiff(image=_convert_np_uint8(rgb_orig),
                     output_path=os.path.join(output_dir, f"original_rgb_t{t}.tiff"),
                     meta=meta_data[t])

        save_geotiff(image=_convert_np_uint8(rgb_pred),
                     output_path=os.path.join(output_dir, f"predicted_rgb_t{t}.tiff"),
                     meta=meta_data[t])

        save_geotiff(image=_convert_np_uint8(rgb_mask),
                     output_path=os.path.join(output_dir, f"masked_rgb_t{t}.tiff"),
                     meta=meta_data[t])


def extract_rgb_imgs(input_img, rec_img, mask_img, channels, mean, std):
    """ Wrapper function to save Geotiff images (original, reconstructed, masked) per timestamp.
    Args:
        input_img: input torch.Tensor with shape (C, T, H, W).
        rec_img: reconstructed torch.Tensor with shape (C, T, H, W).
        mask_img: mask torch.Tensor with shape (C, T, H, W).
        channels: list of indices representing RGB channels.
        mean: list of mean values for each band.
        std: list of std values for each band.
        output_dir: directory where to save outputs.
        meta_data: list of dicts with geotiff meta info.
    """
    rgb_orig_list = []
    rgb_mask_list = []
    rgb_pred_list = []
    
    for t in range(input_img.shape[1]):
        rgb_orig, rgb_pred = process_channel_group(orig_img=input_img[:, t, :, :],
                                                   new_img=rec_img[:, t, :, :],
                                                   channels=channels, data_mean=mean,
                                                   data_std=std)

        rgb_mask = mask_img[channels, t, :, :] * rgb_orig

        # extract images
        rgb_orig_list.append(_convert_np_uint8(rgb_orig))
        rgb_mask_list.append(_convert_np_uint8(rgb_mask))
        rgb_pred_list.append(_convert_np_uint8(rgb_pred))
        
    outputs = rgb_orig_list + rgb_mask_list + rgb_pred_list

    return outputs


def predict_on_images(data_files: list, mask_ratio: float, yaml_file_path: str, checkpoint: str):

    
    try:
        data_files = [x.name for x in data_files]
        print('Path extracted from example')
    except:
        print('Files submitted through UI')

    # Get parameters --------
    print('This is the printout', data_files)

    with open(yaml_file_path, 'r') as f:
        params = yaml.safe_load(f)

    model_params = params["model_args"]
    # data related
    train_params = params["train_params"]
    num_frames = model_params['num_frames']
    img_size = model_params['img_size']
    bands = train_params['bands']
    mean = train_params['data_mean']
    std = train_params['data_std']

    batch_size = 8

    mask_ratio = train_params['mask_ratio'] if mask_ratio is None else mask_ratio
    
    # We must have *num_frames* files to build one example!
    print(f"Expected number of frames: {num_frames}, Actual number of files: {len(data_files)}")

    assert len(data_files) == num_frames, "File list must be equal to expected number of frames."

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using {device} device.\n")

    # Loading data ---------------------------------------------------------------------------------

    input_data, meta_data = load_example(file_paths=data_files, mean=mean, std=std)

    # Create model and load checkpoint -------------------------------------------------------------

    model = MaskedAutoencoderViT(
            **model_params)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--> Model has {total_params:,} parameters.\n")

    model.to(device)

    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from {checkpoint}")

    # Running model --------------------------------------------------------------------------------

    model.eval()
    channels = [bands.index(b) for b in ['blue', 'green', 'red']]  # BGR -> RGB
    
    # Reflect pad if not divisible by img_size
    original_h, original_w = input_data.shape[-2:]
    pad_h = img_size - (original_h % img_size)
    pad_w = img_size - (original_w % img_size)
    input_data = np.pad(input_data, ((0, 0), (0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode='reflect')

    # Build sliding window
    batch = torch.tensor(input_data, device='cpu')
    windows = batch.unfold(3, img_size, img_size).unfold(4, img_size, img_size)
    h1, w1 = windows.shape[3:5]
    windows = rearrange(windows, 'b c t h1 w1 h w -> (b h1 w1) c t h w', h=img_size, w=img_size)

    # Split into batches if number of windows > batch_size
    num_batches = windows.shape[0] // batch_size if windows.shape[0] > batch_size else 1
    windows = torch.tensor_split(windows, num_batches, dim=0)

    # Run model
    rec_imgs = []
    mask_imgs = []
    for x in windows:
        rec_img, mask_img = run_model(model, x, mask_ratio, device)
        rec_imgs.append(rec_img)
        mask_imgs.append(mask_img)

    rec_imgs = torch.concat(rec_imgs, dim=0)
    mask_imgs = torch.concat(mask_imgs, dim=0)

    # Build images from patches
    rec_imgs = rearrange(rec_imgs, '(b h1 w1) c t h w -> b c t (h1 h) (w1 w)',
                         h=img_size, w=img_size, b=1, c=len(bands), t=num_frames, h1=h1, w1=w1)
    mask_imgs = rearrange(mask_imgs, '(b h1 w1) c t h w -> b c t (h1 h) (w1 w)',
                          h=img_size, w=img_size, b=1, c=len(bands), t=num_frames, h1=h1, w1=w1)

    # Cut padded images back to original size
    rec_imgs_full = rec_imgs[..., :original_h, :original_w]
    mask_imgs_full = mask_imgs[..., :original_h, :original_w]
    batch_full = batch[..., :original_h, :original_w]

    # Build RGB images
    for d in meta_data:
        d.update(count=3, dtype='uint8', compress='lzw', nodata=0)

    # save_rgb_imgs(batch[0, ...], rec_imgs_full[0, ...], mask_imgs_full[0, ...],
    #               channels, mean, std, output_dir, meta_data)

    outputs = extract_rgb_imgs(batch_full[0, ...], rec_imgs_full[0, ...], mask_imgs_full[0, ...],
                  channels, mean, std)


    print("Done!")

    return outputs

yaml_file_path = 'Prithvi_100M_config.yaml'
checkpoint = 'checkpoints/Prithvi_100M.pth'

func = partial(predict_on_images, yaml_file_path=yaml_file_path,checkpoint=checkpoint)

def preprocess_example(example_list):
    print('######## preprocessing here ##########')
    example_list = [os.path.join(os.path.abspath(''), x) for x in example_list]
    
    return example_list
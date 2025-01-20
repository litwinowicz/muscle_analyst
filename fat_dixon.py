import nibabel as nib
import numpy as np
import json
import argparse

def process_images(fat_path, water_path, seg_path, output_path, fat_fraction_path, threshold):
    """
    Process fat image using segmentation mask and calculate fat fraction
    
    Args:
        fat_path (str): Path to fat NIfTI file
        water_path (str): Path to water NIfTI file
        seg_path (str): Path to segmentation NIfTI file
        output_path (str): Path to save thresholded image
        fat_fraction_path (str): Path to save fat fraction image
        threshold (int): Threshold for fat voxels (default: 80), values above this threshold will be classified as fat
    """

    fat_img = nib.load(fat_path)
    water_img = nib.load(water_path)
    seg_img = nib.load(seg_path)
    

    fat_data = fat_img.get_fdata()
    water_data = water_img.get_fdata()
    seg_data = seg_img.get_fdata()
    

    binary_mask = seg_data > 0
    masked_fat = fat_data * binary_mask
    masked_water = water_data * binary_mask
    
    # epsilon just in case denominator is 0
    epsilon = 1e-10
    fat_fraction = np.zeros_like(fat_data)
    denominator = masked_fat + masked_water + epsilon
    fat_fraction = np.divide(masked_fat, denominator)

    threshold_data = np.zeros_like(fat_data)
    # 1 for muscle, 2 for fat
    threshold_data[np.logical_and(binary_mask, masked_fat <= threshold)] = 1
    threshold_data[np.logical_and(binary_mask, masked_fat > threshold)] = 2
    
    threshold_img = nib.Nifti1Image(threshold_data, fat_img.affine, fat_img.header)
    fat_fraction_img = nib.Nifti1Image(fat_fraction, fat_img.affine, fat_img.header)
    
    nib.save(threshold_img, output_path)
    nib.save(fat_fraction_img, fat_fraction_path)
    
    masked_ff = fat_fraction[binary_mask]
    stats = {
        'mean_fat_fraction': np.mean(masked_ff),
        'median_fat_fraction': np.median(masked_ff),
        'std_fat_fraction': np.std(masked_ff)
    }
    
    return threshold_img, fat_fraction_img, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process fat/water images and calculate fat fraction.')
    parser.add_argument('--fat', required=True, help='Path to fat NIfTI file')
    parser.add_argument('--water', required=True, help='Path to water NIfTI file')
    parser.add_argument('--segmentation', required=True, help='Path to segmentation NIfTI file')
    parser.add_argument('--output', required=True, help='Output path for thresholded image')
    parser.add_argument('--fat-fraction', required=True, help='Output path for fat fraction image')
    parser.add_argument('--stats', required=True, help='Output path for statistics JSON file')
    # for the patient I've run 80 appears to be a good threshold, but most likely it will not work for everyone
    # the thresholding method might be problematic, but I believe that fat fraction is a better metric than just number of fat voxels anyway
    # so if determining thresholds will turn out to be problematic, it might be a problem that may not be worth solving
    parser.add_argument('--threshold', default=80, help='Threshold for fat voxels (default: 80), values above this threshold will be classified as fat')
    args = parser.parse_args()
    
    threshold_img, fat_fraction_img, stats = process_images(
        args.fat,
        args.water,
        args.segmentation,
        args.output,
        args.fat_fraction,
        int(args.threshold)
    )
    
    with open(args.stats, 'w') as f:
        json.dump(stats, f)
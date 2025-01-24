import nibabel as nib
import numpy as np
import json

# Load files
vert_seg = nib.load(snakemake.input["vertebral_segmentation"])
muscle = nib.load(snakemake.input["muscle_segmentation"] + '/skeletal_muscle.nii.gz')
water_img = nib.load(snakemake.input["water"])
fat_img = nib.load(snakemake.input["fat"])

# Get middle L3 slice
l3_coords = np.where(vert_seg.get_fdata() == 22)
middle_idx = int(np.median(l3_coords[2]))

# Get pixel dimensions and calculate pixel area
pixel_dims = muscle.header.get_zooms()
pixel_area = pixel_dims[0] * pixel_dims[1]  # mm²

# Get the muscle mask for the middle L3 slice
muscle_data = muscle.get_fdata()[:, :, middle_idx].astype(bool)

# Calculate muscle area
area_mm2 = np.sum(muscle_data) * pixel_area
area_cm2 = area_mm2 / 100  # Convert to cm²

# Get water and fat data for the middle L3 slice
water_data = water_img.get_fdata()[:, :, middle_idx]
fat_data = fat_img.get_fdata()[:, :, middle_idx]

# Calculate fat fraction only within the muscle mask
# Add small epsilon to avoid division by zero
epsilon = 1e-10
fat_fraction = fat_data / (water_data + fat_data + epsilon)
masked_fat_fraction = fat_fraction[muscle_data]

# Calculate fat fraction statistics
fat_fraction_stats = {
    "mean_fat_fraction": float(np.mean(masked_fat_fraction)),
    "median_fat_fraction": float(np.median(masked_fat_fraction)),
    "std_fat_fraction": float(np.std(masked_fat_fraction))
}

# Create results dictionary
results = {
    "muscle": area_cm2,
    "fat_fraction_stats": fat_fraction_stats
}

# Save results
with open(snakemake.output["muscle_area"], "w") as f:
    json.dump(results, f, indent=2)
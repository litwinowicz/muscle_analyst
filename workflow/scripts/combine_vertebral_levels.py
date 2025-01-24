import nibabel as nib
import numpy as np
from pathlib import Path


def process_vertebrae(mask_path, level_files_and_labels, output_path):
    """
    Apply binary mask of vertebral bodies to vertebral level segmentations and combine them into a single file
    with specified labels for each level.
    
    Args:
        mask_path: Path to the binary vertebral bodies mask file
        level_files_and_labels: List of tuples [(file_path, label), ...]
        output_path: Path where the combined file will be saved
    """
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata().astype(bool)
    
    combined_data = np.zeros_like(mask_data, dtype=np.uint16)
    
    for file_path, label in level_files_and_labels:
        level_img = nib.load(file_path)
        level_data = level_img.get_fdata()
        
        masked_level = (level_data > 0) & mask_data
        combined_data[masked_level] = label
        
        print(f"Processed {Path(file_path).name} with label {label}")
    
    combined_img = nib.Nifti1Image(combined_data, mask_img.affine, mask_img.header)
    combined_img.to_filename(output_path)
    print(f"Saved combined file to {output_path}")


mask_path = snakemake.input["vertebral_segmentations"][0] + "/vertebrae.nii.gz"
levels_dir = snakemake.input["vertebral_levels_segmentations"][0]
level_files_and_labels = [(levels_dir + f"/vertebrae_{level}.nii.gz", label) for (level, label) in snakemake.params["label_level_pairs"]]
output_path = snakemake.output[0]
    
process_vertebrae(mask_path, level_files_and_labels, output_path)
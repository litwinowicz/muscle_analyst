import nibabel as nib
import numpy as np


def create_ensemble_mask(input_paths, output_path):
    """
    Create an ensemble mask from input NIfTI files.
    
    Parameters:
    input_paths (list): List of paths to input NIfTI mask files
    output_path (str): Path where to save the ensemble mask
    
    Returns:
    str: Path to the created ensemble mask
    """
    # Load all masks
    masks = []
    for path in input_paths:
        nifti_img = nib.load(path)
        masks.append(nifti_img.get_fdata().astype(bool))
    
    # Get reference image for affine and header
    reference_img = nib.load(input_paths[0])
    
    # Create ensemble of all masks
    ensemble = np.logical_or.reduce(masks)
    ensemble_img = nib.Nifti1Image(ensemble.astype(np.uint8),
                                  reference_img.affine,
                                  reference_img.header)
    nib.save(ensemble_img, output_path)
    
    return output_path

if __name__ == "__main__":
    # Example usage
    input_files_all_ensemble = [
        snakemake.input["fat"] + "/skeletal_muscle.nii.gz",
        snakemake.input["water"] + "/skeletal_muscle.nii.gz",
        snakemake.input["ip"] + "/skeletal_muscle.nii.gz",
        snakemake.input["oop"] + "/skeletal_muscle.nii.gz"
    ]
    
    
    
    all_ensemble_path = snakemake.output["all_ensemble"]
    
    create_ensemble_mask(input_files_all_ensemble, all_ensemble_path)

    input_files_water_oop_ensemble = [
        snakemake.input["water"] + "/skeletal_muscle.nii.gz",
        snakemake.input["oop"] + "/skeletal_muscle.nii.gz"
    ]
    water_oop_path = snakemake.output["water_oop_ensemble"]
    create_ensemble_mask(input_files_water_oop_ensemble, water_oop_path)
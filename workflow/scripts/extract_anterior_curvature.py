import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.interpolate import UnivariateSpline
from helper_functions import reorient_to_ras
def trace_anterior_spine_curve_ras(nifti_path_vertebral_levels, nifti_path_discs, output_path=None, smoothing_factor=5.0):
    """
    Trace the anterior curve of the spine in RAS orientation.
    
    Args:
        nifti_path (str): Path to input NIFTI segmentation file
        output_path (str, optional): Path to save the curve as NIFTI file
        smoothing_factor (float): Controls curve smoothness (higher = smoother)
    
    Returns:
        tuple: (anterior_points, smoothed_points, curve_nifti)
    """

    vertebral_levels = nib.load(nifti_path_vertebral_levels)
    discs = nib.load(nifti_path_discs)
    vertebral_levels = reorient_to_ras(vertebral_levels)
    discs = reorient_to_ras(discs)
    vertebral_levels_data = vertebral_levels.get_fdata()
    discs_data = discs.get_fdata()
    data = np.where((discs_data != 0) | (vertebral_levels_data != 0), 1, 0)
    mid_sagittal_idx = data.shape[0] // 2
    mid_slice = data[mid_sagittal_idx, :, :]
    binary_slice = mid_slice > 0

    anterior_points = []
    for col in range(binary_slice.shape[1]):  
        row_points = np.where(binary_slice[:, col])[0]
        if len(row_points) > 0:
            anterior_points.append((row_points[-1], col))
    
    anterior_points = np.array(anterior_points)
    
    if len(anterior_points) == 0:
        raise ValueError("No anterior points found in the segmentation")
    
    sort_idx = np.argsort(anterior_points[:, 1])
    anterior_points = anterior_points[sort_idx]
    
    y_median = ndimage.median_filter(anterior_points[:, 0], size=3)
    
    x = anterior_points[:, 1]  
    spl = UnivariateSpline(x, y_median, k=3, s=smoothing_factor * len(x))
    smoothed_y = spl(x)
    
    smoothed_y = ndimage.gaussian_filter1d(smoothed_y, sigma=2)
    
    curve_volume = np.zeros_like(data)
    
    for i in range(len(x) - 1):
        start = (smoothed_y[i], x[i])  
        end = (smoothed_y[i + 1], x[i + 1])
        
        num_steps = int(np.ceil(np.hypot(end[0] - start[0], end[1] - start[1])))
        if num_steps > 0:
            for t in np.linspace(0, 1, num_steps + 1):
                px = int(round(start[0] * (1 - t) + end[0] * t))
                py = int(round(start[1] * (1 - t) + end[1] * t))
                if 0 <= px < curve_volume.shape[1] and 0 <= py < curve_volume.shape[2]:
                    curve_volume[mid_sagittal_idx, px, py] = 1
    
    curve_nifti = nib.Nifti1Image(curve_volume, vertebral_levels.affine, vertebral_levels.header)
    
    if output_path:
        nib.save(curve_nifti, output_path)
        print(f"Saved curve to: {output_path}")
    
    
    return anterior_points, np.column_stack((smoothed_y, x)), curve_nifti


anterior_points, smoothed_curve, curve_nifti = trace_anterior_spine_curve_ras(snakemake.input["combined_vertebral_levels"], snakemake.input["discs"], snakemake.output[0], smoothing_factor=0)
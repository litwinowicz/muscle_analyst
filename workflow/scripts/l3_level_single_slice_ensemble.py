import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load files
volume = nib.load(snakemake.input["raw_fat"])  # Using water volume as example
ensemble = nib.load(snakemake.input["all_ensemble"])
vert_seg = nib.load(snakemake.input["vertebral_segmentations"])

# Get middle L3 slice
l3_coords = np.where(vert_seg.get_fdata() == 22)
middle_idx = int(np.median(l3_coords[2]))

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(wspace=0.02)

# Get the data for middle L3 slice
volume_data = volume.get_fdata()[:, :, middle_idx]
ensemble_data = ensemble.get_fdata()[:, :, middle_idx].astype(bool)

# Flip data for consistent orientation
volume_data = np.flip(np.flip(volume_data.T, 0), 1)
ensemble_data = np.flip(np.flip(ensemble_data.T, 0), 1)

# Plot raw volume without segmentation
ax1.imshow(volume_data, cmap='gray')
ax1.set_title('Raw Volume')
ax1.set_xticks([])
ax1.set_yticks([])

# Plot volume with ensemble segmentation overlay
ax2.imshow(volume_data, cmap='gray')
masked_ensemble = np.ma.masked_where(~ensemble_data, ensemble_data)
ax2.imshow(masked_ensemble, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
ax2.set_title('Volume with Ensemble Segmentation')
ax2.set_xticks([])
ax2.set_yticks([])

# Save visualization
plt.savefig(snakemake.output["simple_viz"], 
            format='png', 
            bbox_inches='tight', 
            dpi=300)
plt.close()
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import calculate_fat_fraction

fat = nib.load(snakemake.input["raw_fat"])
water = nib.load(snakemake.input["raw_water"])
ip = nib.load(snakemake.input["raw_ip"])
oop = nib.load(snakemake.input["raw_oop"])
volumes = [fat, water, ip, oop]

vert_seg = nib.load(snakemake.input["vertebral_segmentations"])
fat_muscle = nib.load(snakemake.input["muscle_fat"] + '/skeletal_muscle.nii.gz')
water_muscle = nib.load(snakemake.input["muscle_water"] + '/skeletal_muscle.nii.gz')
ip_muscle = nib.load(snakemake.input["muscle_ip"] + '/skeletal_muscle.nii.gz')
oop_muscle = nib.load(snakemake.input["muscle_oop"] + '/skeletal_muscle.nii.gz')
water_oop_ensemble = nib.load(snakemake.input["water_oop_ensemble"])
all_ensemble = nib.load(snakemake.input["all_ensemble"])
muscles = [fat_muscle, water_muscle, ip_muscle, oop_muscle, water_oop_ensemble, all_ensemble]
height = snakemake.params["height"]
print("height", height)

l3_coords = np.where(vert_seg.get_fdata() == 22)
middle_idx = int(np.median(l3_coords[2]))

fig1, axs = plt.subplots(5, 7, figsize=(35, 25))
plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0.02, right=0.98, top=0.98, bottom=0.02)

for i, vol in enumerate(volumes):
    vol_data = vol.get_fdata()[:, :, middle_idx]
    vol_data = np.flip(np.flip(vol_data.T, 0), 1)
    
    ax = axs[i, 0]
    ax.imshow(vol_data, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    if i == 0:
        ax.set_title('Raw Volume')
    ax.set_ylabel(f'{["Fat", "Water", "IP", "OOP"][i]} volume')

for i, vol in enumerate(volumes):
    vol_data = vol.get_fdata()[:, :, middle_idx]
    vol_data = np.flip(np.flip(vol_data.T, 0), 1)
    
    for j, muscle in enumerate(muscles):
        muscle_data = muscle.get_fdata()[:, :, middle_idx].astype(bool)
        muscle_data = np.flip(np.flip(muscle_data.T, 0), 1)
        
        ax = axs[i, j+1]  # +1 because first column is for raw volumes
        ax.imshow(vol_data, cmap='gray')
        masked_muscle = np.ma.masked_where(~muscle_data, muscle_data)
        ax.imshow(masked_muscle, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        if i == 0:
            ax.set_title(f'{["Fat", "Water", "IP", "OOP", "Water-OOP Ensemble", "All Ensemble"][j]} muscle')

axs[4, 0].remove()
ax_invisible = fig1.add_subplot(5, 7, 29) 
ax_invisible.set_visible(False)
ax_invisible.set_ylabel('Muscle mask')


for i, muscle in enumerate(muscles):
    muscle_data = muscle.get_fdata()[:, :, middle_idx].astype(bool)
    muscle_data = np.flip(np.flip(muscle_data.T, 0), 1)
    
    ax = axs[4, i+1] 
    masked_muscle = np.ma.masked_where(~muscle_data, muscle_data)
    ax.imshow(masked_muscle, cmap='Reds', vmin=0, vmax=1)
    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks([])

pixel_dims = muscles[0].header.get_zooms()
pixel_area = pixel_dims[0] * pixel_dims[1]  

muscle_areas = []
muscle_names = ["Fat", "Water", "IP", "OOP", "Water-OOP Ensemble", "All Ensemble"]
for i, muscle in enumerate(muscles):
    muscle_data = muscle.get_fdata()[:, :, middle_idx].astype(bool)
    area_mm2 = np.sum(muscle_data) * pixel_area
    area_cm2 = area_mm2 / 100
    area_cm2_normalized = area_cm2 / (height * height)
    muscle_areas.append(area_cm2_normalized)

for i, area in enumerate(muscle_areas):
    ax = axs[4, i+1]  
    ax.text(0.5, -0.1, f'Area normalized to height: {area:.2f} cm²/m²',
            horizontalalignment='center',
            transform=ax.transAxes)

plt.savefig(snakemake.output["segmentation"], format='pdf', bbox_inches='tight', dpi=300)
plt.close()

fig2, axs = plt.subplots(2, 3, figsize=(20, 12))
plt.subplots_adjust(wspace=0.02)

fat_data = fat.get_fdata()[:, :, middle_idx]
water_data = water.get_fdata()[:, :, middle_idx]

segmentations = [fat_muscle, water_oop_ensemble, all_ensemble]
titles = ['Original Segmentation', 'Water-OOP Ensemble', 'All Ensemble']

for i, (seg, title) in enumerate(zip(segmentations, titles)):
    muscle_data = seg.get_fdata()[:, :, middle_idx].astype(bool)
    fat_fraction, stats = calculate_fat_fraction(fat_data, water_data, muscle_data)

    fat_data_flip = np.flip(np.flip(fat_data.T, 0), 1)
    fat_fraction = np.flip(np.flip(fat_fraction.T, 0), 1)
    muscle_data = np.flip(np.flip(muscle_data.T, 0), 1)

    row = i // 3
    col = i % 3
    ax = axs[row, col]
    
    ax.imshow(fat_data_flip, cmap='gray')
    masked_ff = np.ma.masked_where(~muscle_data, fat_fraction)
    im = ax.imshow(masked_ff, cmap='hot', alpha=0.7, vmin=0, vmax=1)
    ax.set_title(f'{title}\nMean FF: {stats["mean_fat_fraction"]:.3f}')
    ax.set_xticks([])
    ax.set_yticks([])

# Add colorbar
cbar_ax = fig2.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = plt.colorbar(im, cax=cbar_ax)
cbar.set_label('Fat Fraction')

# Save second visualization
plt.savefig(snakemake.output["fat_fraction"], 
            format='pdf', bbox_inches='tight', dpi=300)
plt.close()
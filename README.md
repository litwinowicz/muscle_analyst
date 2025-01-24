# Medical Image Segmentation Pipeline

This Snakemake pipeline performs automated segmentation and analysis of medical images, specifically focusing on muscle and vertebrae segmentation using the TotalSegmentator tool.

## Prerequisites

- Snakemake
- Singularity/Docker
- TotalSegmentator license (specified in config as `TOTALSEGMENTATOR_LICENSE`)
- RMarkdown



## Input Data Structure

Place your input data in the following structure:
```
resources/
├── metadata.csv
└── {subject_id}/
    ├── water.nii.gz
    ├── fat.nii.gz
    ├── ip.nii.gz
    └── oop.nii.gz
```

The `metadata.csv` file should contain at least the following columns:
- subject_id (str)
- height

## Pipeline Overview

The pipeline performs the following main tasks:

1. **Muscle Segmentation**
   - Standard tissue segmentation
   - 4-type tissue segmentation
   - Vertebrae level segmentation
   - Combined vertebrae segmentation

2. **Ensemble Creation**
   - Creates ensemble masks from multiple sequences
   - Generates water-OOP and all-sequence ensembles

3. **Analysis**
   - Calculates muscle area at mid-L3 level
   - Measures volumes between L1 and L3
   - Generates fat fraction maps
   - Creates visualization plots

## Main Commands

- `snakemake -c 1 --use-conda --use-singularity -p "results/tmp/calculate_all_muscle_segmentations_for_all_patients.txt"`: Runs muscle segmentation for all patients and all sequences
- `snakemake -c 1 --use-conda --use-singularity -p "results/tmp/calculate_vertebral_segmentations_for_all_patients.txt"`: Runs vertebral segmentation for all patients
- `snakemake -c all --use-conda --use-singularity -p "results/final/analysis"`: Generates results for all patient (plots and general analysis report)

I suggest running the first two commands first with one core and then the third one with all cores - TS did not work great with multithreading for me.

## Output Structure

The pipeline generates results in the following structure:
```
results/
├── final/
│   ├── analysis/
│   ├── segmentation_plots/
│   ├── fat_fraction_plots/
│   └── single_slice_ensemble_plots/
└── volumes_and_areas.csv
```

Intermediate results are stored for each subject and sequence in the `results/{subject_id}/{sequence}` directory. Currently they are not removed but when the pipeline is ready they should be removed or at least we should have a flag to remove them.

## Usage

1. Ensure your input data is properly organized in the `resources/` directory
2. Configure your TotalSegmentator license in the Snakefile
3. Run the pipeline:
   ```
   snakemake -c 1 --use-conda --use-singularity -p "results/tmp/calculate_all_muscle_segmentations_for_all_patients.txt"
   snakemake -c 1 --use-conda --use-singularity -p "results/tmp/calculate_vertebral_segmentations_for_all_patients.txt"
   snakemake -c all --use-conda --use-singularity -p "results/final/analysis"
   ```
## Key Output Files

- `results/final/analysis.pdf`: Complete analysis report
- `results/final/segmentation_plots`: Segmentation visualizations for each patient and each sequence
- `results/final/fat_fraction_plots`: Fat fraction visualizations for each patient
- `results/final/single_slice_ensemble_plots`: Single slice ensemble visualizations for each patient
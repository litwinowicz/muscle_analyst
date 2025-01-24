import json
import pandas as pd
import re

def extract_subject_and_sequence(filepath):
    # Extract subject_id and sequence from filepath
    match = re.search(r'results/(\d+)/(\w+)/', filepath)
    if match:
        return match.group(1), match.group(2)
    return None, None

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def process_files(volume_files, area_files, metadata_df):
    results = []
    
    # Create a mapping of (subject_id, sequence) to filepaths
    volume_map = {extract_subject_and_sequence(f): f for f in volume_files}
    area_map = {extract_subject_and_sequence(f): f for f in area_files}
    
    # Process each subject and sequence combination
    for _, row in metadata_df.iterrows():
        subject_id = row['subject_id']
        height = row['height']
        
        for sequence in ['water', 'fat', 'ip', 'oop', 'all_ensemble', 'water_oop_ensemble']:
            # Initialize all values to None
            data = {
                'subject_id': subject_id,
                'height': height,
                'sequence': sequence,
                'muscle_area': None,
                'original_volume': None,
                'cropped_volume': None,
                'volume_mean_fat_fraction': None,
                'volume_median_fat_fraction': None,
                'volume_std_fat_fraction': None,
                'area_mean_fat_fraction': None,
                'area_median_fat_fraction': None,
                'area_std_fat_fraction': None
            }
            
            # Get volume data
            volume_file = volume_map.get((subject_id, sequence))
            if volume_file:
                try:
                    volume_data = load_json(volume_file)
                    data['original_volume'] = volume_data['original_volume_cm3']
                    data['cropped_volume'] = volume_data['cropped_volume_cm3']
                    
                    # Extract volume fat fraction stats
                    if 'fat_fraction_stats' in volume_data:
                        fat_stats = volume_data['fat_fraction_stats']
                        data['volume_mean_fat_fraction'] = fat_stats.get('mean_fat_fraction')
                        data['volume_median_fat_fraction'] = fat_stats.get('median_fat_fraction')
                        data['volume_std_fat_fraction'] = fat_stats.get('std_fat_fraction')
                except (FileNotFoundError, KeyError) as e:
                    print(f"Warning: Error processing volume file {volume_file}: {e}")
            
            # Get area data
            area_file = area_map.get((subject_id, sequence))
            if area_file:
                try:
                    area_data = load_json(area_file)
                    data['muscle_area'] = area_data['muscle']
                    
                    # Extract area fat fraction stats
                    if 'fat_fraction_stats' in area_data:
                        fat_stats = area_data['fat_fraction_stats']
                        data['area_mean_fat_fraction'] = fat_stats.get('mean_fat_fraction')
                        data['area_median_fat_fraction'] = fat_stats.get('median_fat_fraction')
                        data['area_std_fat_fraction'] = fat_stats.get('std_fat_fraction')
                except (FileNotFoundError, KeyError) as e:
                    print(f"Warning: Error processing area file {area_file}: {e}")
            
            results.append(data)
    
    return pd.DataFrame(results)

# Get input files from snakemake
volume_files = snakemake.input["volumes"]
area_files = snakemake.input["areas"]
metadata_df = snakemake.params["metadata"]

# Process all files
results_df = process_files(volume_files, area_files, metadata_df)

# Save to CSV
csv_output = str(snakemake.output["csv"])
results_df.to_csv(csv_output, index=False)

# Save to JSON as required by snakemake
results_dict = results_df.to_dict(orient='records')
with open(snakemake.output["json"], 'w') as f:
    json.dump(results_dict, f, indent=2)
import os
import json
import shutil


# Fix the filename so that the filename is the same as the caption_idx.
def copy_and_rename_json_files(source_dir, target_dir):
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Iterate over all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(source_dir, filename)
            
            try:
                # Load the JSON file
                with open(file_path, 'r') as file:
                    data = json.load(file)
                
                # Skip if the JSON file is empty or doesn't have the expected structure
                if not data or not isinstance(data, dict):
                    print(f"Skipping {filename}: Empty or invalid structure")
                    continue
                
                # Extract the first caption_idx from the first image entry
                first_image_data = next(iter(data.values()))
                if not isinstance(first_image_data, dict):
                    print(f"Skipping {filename}: Invalid nested structure")
                    continue
                
                caption_idx = next(iter(first_image_data.keys()))
                
                # New file name
                new_filename = f"{caption_idx}.json"
                new_file_path = os.path.join(target_dir, new_filename)
                
                # Copy the file to the target directory with the new name
                shutil.copy(file_path, new_file_path)
                print(f"Copied and renamed {filename} to {new_filename} in {target_dir}")
            
            except json.JSONDecodeError:
                print(f"Skipping {filename}: Invalid JSON format")
            except Exception as e:
                print(f"Skipping {filename}: Unexpected error - {e}")



# Remove `product` data from `car` data.
def remove_noisy_data(target_dir):
    for filename in os.listdir(target_dir):
        # Check if the file is a JSON file
        if filename.endswith(".json"):
            filepath = os.path.join(target_dir, filename)
            
            try:
                # Open and load the JSON file
                with open(filepath, "r") as file:
                    data = json.load(file)
                
                # Check the number of keys in the JSON file
                if isinstance(data, dict) and len(data.keys()) == 11293:
                    # Remove the file if the condition is met
                    os.remove(filepath)
                    print(f"Removed: {filename} (Number of keys: 11293)")
            
            except json.JSONDecodeError:
                print(f"Skipping {filename}: Invalid JSON format")
            except Exception as e:
                print(f"Error processing {filename}: {e}")



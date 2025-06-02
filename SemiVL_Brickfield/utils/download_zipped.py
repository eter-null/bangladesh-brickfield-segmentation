import shutil
import os

# Ask user for folder path
folder_path = input("Enter the full folder path to zip: ").strip()

# Check if the folder exists
if not os.path.isdir(folder_path):
    print("The specified folder does not exist.")
else:
    # Create an output zip path based on folder name
    base_folder_name = os.path.basename(folder_path.rstrip("/"))
    parent_folder = os.path.dirname(folder_path)
    output_zip_path = os.path.join(parent_folder, f"{base_folder_name}_zipped")

    # Create zip archive
    shutil.make_archive(output_zip_path, 'zip', folder_path)

    print(f"Zipped folder to: {output_zip_path}.zip")

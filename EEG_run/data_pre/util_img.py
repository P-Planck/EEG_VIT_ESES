import os
import shutil

# Function to extract and copy images from multiple subfolders
def extract_images(base_path,start_idx, num_folders, output_folder):
    # Create destination directories for 'n' and 'y'
    n_output_folder = os.path.join(output_folder, 'n')
    y_output_folder = os.path.join(output_folder, 'y')
    os.makedirs(n_output_folder, exist_ok=True)
    os.makedirs(y_output_folder, exist_ok=True)

    # Loop through each folder
    for folder_idx in range(start_idx, num_folders + 1):
        print(f"processing idx {folder_idx}")
        folder_path = os.path.join(base_path, str(folder_idx))


        # Paths to the 'n' and 'y' subfolders in each folder
        n_folder = os.path.join(folder_path, 'n')
        y_folder = os.path.join(folder_path, 'y')
        
        # Check if the 'n' folder exists, and copy all images to the final 'n' folder
        if os.path.exists(n_folder):
            for file_name in os.listdir(n_folder):
                if file_name.endswith('.jpg'):
                    src_file = os.path.join(n_folder, file_name)
                    dst_file = os.path.join(n_output_folder, f'{folder_idx}_{file_name}')  # Prefix with folder number
                    shutil.copy(src_file, dst_file)

        # Check if the 'y' folder exists, and copy all images to the final 'y' folder
        if os.path.exists(y_folder):
            for file_name in os.listdir(y_folder):
                if file_name.endswith('.jpg'):
                    src_file = os.path.join(y_folder, file_name)
                    dst_file = os.path.join(y_output_folder, f'{folder_idx}_{file_name}')  # Prefix with folder number
                    shutil.copy(src_file, dst_file)
    

# Define the base path where the EEG data is stored and the output folder for EEG_IMG
base_path = r'./EEG_data/EEG_IMG_sinv_pivot'
output_folder = r'./EEG_data/EEG_IMG_sinv_pivot_all'
num_folders = 30  # Number of folders (from 1 to 30)
start_idx = 1
# Extract images from all subfolders into EEG_IMG/n and EEG_IMG/y
extract_images(base_path,start_idx, num_folders, output_folder)

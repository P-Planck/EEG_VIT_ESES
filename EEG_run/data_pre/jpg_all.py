import os
import mne
import numpy as np
import pandas as pd
from PIL import Image

# Function to read EDF file and convert it to a DataFrame
def edf_to_dataframe(edf_file):
    # Read the EDF file using mne with encoding set to 'latin1'
    raw = mne.io.read_raw_edf(edf_file, preload=True, stim_channel=None, encoding='latin1')
    
    # Get the data and times
    data, times = raw[:]
    
    # Create a DataFrame with the data, using channel names as column headers
    df = pd.DataFrame(data.T, columns=raw.ch_names)
    
    return df, len(raw.ch_names)  # Return the DataFrame and the number of channels

# Function to convert grayscale matrix to RGB
def grayscale_to_rgb(matrix):
    return np.stack([matrix] * 3, axis=-1)

# Function to normalize the channel data
def normalize_channel(channel):
    channel_min = np.min(channel)
    channel_ptp = np.ptp(channel)  # Peak-to-peak (max - min)
    
    if channel_ptp == 0:
        # If all values are the same, set it to mid-gray (127)
        return np.full_like(channel, 127, dtype=np.uint8)
    
    # Normalize to the range 0-255
    return ((channel - channel_min) / channel_ptp * 255).astype(np.uint8)

# Main function to process data from multiple directories
def process_eeg_data(base_path, num_folders,start_idx):
    for folder_idx in range(start_idx, num_folders + 1):
        folder_path = os.path.join(base_path, str(folder_idx))
        
        # Generate the file names dynamically based on folder index
        edf_file = os.path.join(folder_path, f'{folder_idx}.edf')
        excel_file = os.path.join(folder_path, f'{folder_idx}.xlsx')
        
        # Load EDF file and convert to DataFrame
        df, num_channels = edf_to_dataframe(edf_file)
        
        # Create directories for storing images
        y_folder = os.path.join(folder_path, 'y')
        n_folder = os.path.join(folder_path, 'n')
        os.makedirs(y_folder, exist_ok=True)
        os.makedirs(n_folder, exist_ok=True)
        
        # Load labels from Excel
        labels_df = pd.read_excel(excel_file)
        labels = labels_df.iloc[:, 0].values  # Assuming labels are in the first column
        
        # Iterate through the EEG data and save images
        num_samples = 3599  # We know there are 3600 samples and labels
        for i in range(num_samples):
            # Extract data for the current sample (num_channels x 500 points)
            data = df.iloc[i*500:(i+1)*500, :num_channels].to_numpy().T  # Transpose to get (num_channels x 500)
            
            # Normalize data to 0-255 for grayscale image
            grayscale_image = np.array([normalize_channel(row) for row in data])
            
            # Ensure the image shape is correct: (num_channels, 500)
            if grayscale_image.shape != (num_channels, 500):
                print(f"Warning: Unexpected shape for sample {i+1} in folder {folder_idx}: {grayscale_image.shape}")
            
            # Convert grayscale to RGB
            rgb_image = grayscale_to_rgb(grayscale_image)
            
            # Convert to PIL image
            img = Image.fromarray(rgb_image)
            
            # Save the image in the respective folder based on the label
            label = labels[i]  # Access the label correctly
            folder = y_folder if label == 1 else n_folder
            img.save(os.path.join(folder, f'image_{i+1}.jpg'))

# Define base path and number of folders
base_path = r'G:\EEG\EEG_data'
num_folders = 30
start_idx = 10

# Process all EEG data
process_eeg_data(base_path, num_folders,start_idx)

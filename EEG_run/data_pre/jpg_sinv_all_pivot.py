import os
import mne
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Function to read EDF file and convert it to a DataFrame
def edf_to_dataframe(edf_file):
    raw = mne.io.read_raw_edf(edf_file, preload=True, stim_channel=None, encoding='latin1')
    data, times = raw[:]
    df = pd.DataFrame(data.T, columns=raw.ch_names)
    return df, len(raw.ch_names)

# Function to normalize the channel data
def normalize_channel(channel):
    channel_min = np.min(channel)
    channel_ptp = np.ptp(channel)  # Peak-to-peak (max - min)
    if channel_ptp == 0:
        return np.full_like(channel, 127, dtype=np.uint8)  # Mid-gray
    return ((channel - channel_min) / channel_ptp * 255).astype(np.uint8)

# Main function to process data from multiple directories
def process_eeg_data(base_input_path,base_output_path, num_folders, start_idx, csv_file, num_categories):
    # Read CSV for the prediction values
    csv_data = pd.read_csv(csv_file)
    csv_data['folder'] = csv_data['image'].str.extract(r'(\d+)_image_\d+').astype(int)  # Extract folder index
    csv_data['index'] = csv_data['image'].str.extract(r'\d+_image_(\d+)').astype(int)  # Extract image index

    # Group by folder for efficient processing
    grouped_data = csv_data.groupby('folder')

    # create base_output_path
    if not os.path.exists(base_output_path):
        os.makedirs(base_output_path)
        print(f"directory {base_output_path} created")
    else:
        print(f"directory {base_output_path} already exists")

    for folder_idx in tqdm(range(start_idx, num_folders + 1), desc="Processing folders"):
        input_folder_path = os.path.join(base_input_path, str(folder_idx))
        output_folder_path = os.path.join(base_output_path, str(folder_idx))
        edf_file = os.path.join(input_folder_path, f'{folder_idx}.edf')
        excel_file = os.path.join(input_folder_path, f'{folder_idx}.xlsx')

        # Load EDF file and convert to DataFrame
        df, num_channels = edf_to_dataframe(edf_file)

        # Create directories for storing images
        y_folder = os.path.join(output_folder_path, 'y')
        n_folder = os.path.join(output_folder_path, 'n')
        os.makedirs(y_folder, exist_ok=True)
        os.makedirs(n_folder, exist_ok=True)

        # Load labels from Excel
        labels_df = pd.read_excel(excel_file)
        labels = labels_df.iloc[:, 0].values

        # Process samples in the current folder based on CSV
        if folder_idx in grouped_data.groups:
            folder_data = grouped_data.get_group(folder_idx)

            for _, row in tqdm(folder_data.iterrows(), desc=f"Processing samples in folder {folder_idx}", total=len(folder_data), leave=False):
                img_idx = row['index']
                csv_value = row['result']

                # Extract data for the current sample
                data = df.iloc[(img_idx - 1) * 500:img_idx * 500, :num_channels].to_numpy().T
                grayscale_image = np.array([normalize_channel(row) for row in data])

                if grayscale_image.shape != (num_channels, 500):
                    print(f"Warning: Unexpected shape for sample {img_idx} in folder {folder_idx}: {grayscale_image.shape}")

                # Prepare the three channels
                grayscale_channel = grayscale_image
                sine_value = np.sin(csv_value / num_categories)
                sine_channel = np.full(grayscale_channel.shape, sine_value, dtype=np.float32)
                zero_channel = np.zeros_like(grayscale_channel, dtype=np.float32)

                label = labels[img_idx - 1]
                if label == 1:


                # Stack the channels
                    stacked_image = np.stack([grayscale_channel, sine_channel, zero_channel], axis=-1)

                if label ==0:
                    stacked_image = np.stack([sine_channel, grayscale_channel,  zero_channel], axis=-1)

                # Convert to PIL image and save
                img = Image.fromarray((stacked_image * 255).astype(np.uint8))

                folder = y_folder if label == 1 else n_folder
                img.save(os.path.join(folder, f'image_{img_idx}.jpg'))

# Define parameters
base_input_path = './EEG_data/EEG_IMG_BASE'
base_output_path = './EEG_data/EEG_IMG_sinv_pivot'
num_folders = 30
start_idx = 1
csv_file = 'prediction_results.csv'  # Path to CSV file with prediction results
num_categories = 3  # Number of categories in CSV

# Process the EEG data
process_eeg_data(base_input_path,base_output_path, num_folders, start_idx, csv_file, num_categories)

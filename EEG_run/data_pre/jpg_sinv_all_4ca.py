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
def process_eeg_data(base_input_path, base_output_path, num_folders, start_idx, csv_file):
    # Read CSV for the prediction values
    csv_data = pd.read_csv(csv_file)
    csv_data['folder'] = csv_data['image'].str.extract(r'(\d+)_image_\d+').astype(int)  # Extract folder index
    csv_data['index'] = csv_data['image'].str.extract(r'\d+_image_(\d+)').astype(int)  # Extract image index

    # Group by folder for efficient processing
    grouped_data = csv_data.groupby('folder')

    # Create base_output_path and subfolders
    categories = ['yy', 'nn', 'yn', 'ny']
    num_categories = len(categories)
    print(f"num_categories: {num_categories}")


    for folder_idx in tqdm(range(start_idx, num_folders + 1), desc="Processing folders"):
        input_folder_path = os.path.join(base_input_path, str(folder_idx))
        output_folder_path = os.path.join(base_output_path, str(folder_idx))
        edf_file = os.path.join(input_folder_path, f'{folder_idx}.edf')
        excel_file = os.path.join(input_folder_path, f'{folder_idx}.xlsx')
        for category in categories:
            category_path = os.path.join(output_folder_path, category)
            os.makedirs(category_path, exist_ok=True)
        # Load EDF file and convert to DataFrame
        df, num_channels = edf_to_dataframe(edf_file)

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
                grayscale_channel = grayscale_image
                sine_value = np.sin(csv_value / num_categories)
                sine_channel = np.full(grayscale_channel.shape, sine_value, dtype=np.float32)
                zero_channel = np.zeros_like(grayscale_channel, dtype=np.float32)

                # Stack the channels
                stacked_image = np.stack([grayscale_channel, sine_channel, zero_channel], axis=-1)

                # Convert to PIL image and save
                img = Image.fromarray((stacked_image * 255).astype(np.uint8))


                label = labels[img_idx - 1]


                # Determine folder based on label and csv_value
                if label == 1 and csv_value == 0:
                    folder = 'yy'
                elif label == 0 and csv_value == 0:
                    folder = 'nn'
                elif label == 1 and csv_value == 1:
                    folder = 'yn'
                elif label == 0 and csv_value == 2:
                    folder = 'ny'
                else:
                    print(f"Unexpected combination: Label={label}, CSV={csv_value}")
                    continue

                ## directly store the images in folder of several folders to train
                # save_path = os.path.join(base_output_path, folder, f'{folder_idx}_image_{img_idx}.jpg')


                ## separately store the images in output_folder_path

                save_path = os.path.join(output_folder_path, folder, f'{folder_idx}_image_{img_idx}.jpg')
                img.save(save_path)

# Define parameters
base_input_path = './EEG_data/EEG_IMG_BASE'
base_output_path = './EEG_data/EEG_IMG_4c_sep'
num_folders = 10
start_idx = 1
csv_file = 'prediction_results.csv'  # Path to CSV file with prediction results

# Process the EEG data
process_eeg_data(base_input_path, base_output_path, num_folders, start_idx, csv_file)
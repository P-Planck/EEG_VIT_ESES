import os
import mne
import numpy as np
import pandas as pd
from PIL import Image

# Your function to read EDF file and convert it to a DataFrame
def edf_to_dataframe(edf_file):
    # Read the EDF file using mne with encoding set to 'latin1'
    raw = mne.io.read_raw_edf(edf_file, preload=True, stim_channel=None, encoding='latin1')
    
    # Print basic information about the file
    print("EDF File Info:")
    print(raw.info)
    
    # Print the channel names
    print("\nChannel Names:")
    for name in raw.ch_names:
        print(name)
    
    # Print additional header information
    print("\nAdditional Header Information:")
    print(f"Number of Channels: {len(raw.ch_names)}")
    print(f"Sampling Frequency: {raw.info['sfreq']} Hz")
    print(f"Measurement Date: {raw.info['meas_date']}")
    print(f"Highpass Filter: {raw.info['highpass']} Hz")
    print(f"Lowpass Filter: {raw.info['lowpass']} Hz")
    
    # Print channel-specific information
    print("\nChannel-specific Information:")
    for ch in raw.info['chs']:
        print(f"Name: {ch['ch_name']}, Type: {ch['kind']}, Unit: {ch['unit']}, Range: {ch['range']}")
    
    # Get the data and times
    data, times = raw[:]
    
    # Create a DataFrame with the data
    df = pd.DataFrame(data.T, columns=raw.ch_names)
    
    return df

# Load EDF file and convert to DataFrame
edf_path = r'E:\EEG\1\1.edf'
df = edf_to_dataframe(edf_path)

# Create directories for storing images
y_folder = r'E:\EEG\1\y'
n_folder = r'E:\EEG\1\n'
os.makedirs(y_folder, exist_ok=True)
os.makedirs(n_folder, exist_ok=True)

# Load labels from Excel
excel_path = r'E:\EEG\1\1.xlsx'
labels_df = pd.read_excel(excel_path)
labels = labels_df.iloc[:, 0].values  # Assuming labels are in the first column

# Function to convert grayscale matrix to RGB
def grayscale_to_rgb(matrix):
    return np.stack([matrix] * 3, axis=-1)

def normalize_channel(channel):
    channel_min = np.min(channel)
    channel_ptp = np.ptp(channel)  # Peak-to-peak (max - min)
    
    if channel_ptp == 0:
        # If all values are the same, set it to mid-gray (127)
        return np.full_like(channel, 127, dtype=np.uint8)
    
    # Normalize to the range 0-255
    return ((channel - channel_min) / channel_ptp * 255).astype(np.uint8)

# # Iterate through the EEG data and save images
# num_samples = 3600  # We know there are 3600 samples and labels
# for i in range(num_samples):
#     # Extract data for the current sample (40 channels x 500 points)
#     data = df.iloc[i*500:(i+1)*500, :40].to_numpy().T
    
#     # Normalize data to 0-255 for grayscale image
#     grayscale_image = (255 * (data - np.min(data)) / np.ptp(data)).astype(np.uint8)
    
#     # Convert grayscale to RGB
#     rgb_image = grayscale_to_rgb(grayscale_image)
    
#     # Convert to PIL image
#     img = Image.fromarray(rgb_image)
    
#     # Save the image in the respective folder based on the label
#     label = labels[i]  # Now we are sure there are 3600 labels
#     folder = y_folder if label == 1 else n_folder
#     img.save(os.path.join(folder, f'image_{i+1}.jpg'))

num_samples = 3600  # We know there are 3600 samples and labels
for i in range(num_samples):
    # Extract data for the current sample (40 channels x 500 points)
    # Ensure that the matrix is 40 (channels) x 500 (time points)
    data = df.iloc[i*500:(i+1)*500, :40].to_numpy().T  # Transpose to get 40x500
    # print(data)

    # Normalize data to 0-255 for grayscale image
    # grayscale_image = (255 * (data - np.min(data)) / np.ptp(data)).astype(np.uint8)
    grayscale_image = np.array([normalize_channel(row) for row in data])
    # print("grayscale_image",grayscale_image)

    # Ensure the image shape is correct: 40x500
    if grayscale_image.shape != (40, 500):
        print(f"Warning: Unexpected shape for sample {i+1}: {grayscale_image.shape}")
    
    # Convert grayscale to RGB
    rgb_image = grayscale_to_rgb(grayscale_image)  # Convert 40x500 grayscale to RGB
    print("rgb\n",rgb_image)
    # Convert to PIL image
    img = Image.fromarray(rgb_image)
    
    # Save the image in the respective folder based on the label
    label = labels[i]  # Access the label correctly
    folder = y_folder if label == 1 else n_folder
    img.save(os.path.join(folder, f'image_{i+1}.jpg'))

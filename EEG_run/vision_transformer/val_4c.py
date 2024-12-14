import os
import json
import time
import random
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm  # Progress bar

from vit_model import vit_base_patch16_224_in21k as create_model


def select_random_images(folder_path, num_images=1000):
    # Select all images with .jpg extension in the folder
    images = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith('.jpg')]
    # Randomly select 'num_images' from the list
    return random.sample(images, min(num_images, len(images)))  # Handle case where folder has fewer images


def process_image(img_path, data_transform):
    img = Image.open(img_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)  # Add batch dimension
    return img


def main():
    Validation_folder = "Validation_result_4c"
    os.makedirs(Validation_folder, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}")

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Folders for 4 classes
    base_folder = './EEG_data/EEG_IMG_4c'
    classes = ['nn', 'ny', 'yn', 'yy']
    folder_paths = {cls: os.path.join(base_folder, cls) for cls in classes}

    # Randomly select 1000 images for each class
    selected_images = {cls: select_random_images(folder_paths[cls], 1000) for cls in classes}

    # Read class indices
    json_path = './class_indices_4c.json'
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # Create model and load weights
    model = create_model(num_classes=len(classes), has_logits=False).to(device)
    model_weight_path = "VIT_sinv_4c/epoch25.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    true_labels = []
    predicted_probs = []
    predicted_classes = []

    # File to save the results
    result_file = f'{Validation_folder}/{time.time()}_classification_results_4c.txt'

    with open(result_file, 'w') as f:
        f.write("Image Path, True Label, Predicted Class, Predicted Probabilities\n")

        for true_label, cls in enumerate(classes):
            for img_path in tqdm(selected_images[cls], desc=f"Processing class '{cls}' images"):
                img = process_image(img_path, data_transform)
                with torch.no_grad():
                    output = torch.squeeze(model(img.to(device))).cpu()
                    predict = torch.softmax(output, dim=0)
                    predicted_probs.append(predict.numpy())
                    predicted_class = torch.argmax(predict).item()
                    predicted_classes.append(predicted_class)
                    true_labels.append(true_label)

                    # Write to file
                    f.write(f"{img_path}, {true_label}, {predicted_class}, {predict.numpy()}\n")

    # Convert lists to arrays for easier handling
    predicted_probs = np.array(predicted_probs)
    predicted_classes = np.array(predicted_classes)
    true_labels = np.array(true_labels)

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (4 Classes)')
    plt.show()


if __name__ == '__main__':
    main()
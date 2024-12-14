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
    return random.sample(images, num_images)

def process_image(img_path, data_transform):
    img = Image.open(img_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)  # Add batch dimension
    return img

def main():
    Validation_folder = "Validation_result"
    os.makedirs(Validation_folder, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device is {device}")
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Folders for 'n' and 'y'
    n_folder = './EEG_data/EEG_IMG/n'
    y_folder = './EEG_data/EEG_IMG/y'

    # Randomly select 1000 images from each folder
    n_images = select_random_images(n_folder, 1000)
    y_images = select_random_images(y_folder, 1000)

    # Read class indices
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # Create model and load weights
    model = create_model(num_classes=2, has_logits=False).to(device)
    model_weight_path = "EEG_run/weights/model-80-3stack.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    true_labels = []
    predicted_probs = []
    predicted_classes = []

    # File to save the results
    result_file = f'{Validation_folder}/{time.time()}_classification_results.txt'

    with open(result_file, 'w') as f:
        f.write("Image Path, True Label, Predicted Class, Predicted Probabilities (n, y)\n")
        
        # Predict for 'n' class (label 0)
        for img_path in tqdm(n_images, desc="Processing class 'n' images"):
            img = process_image(img_path, data_transform)
            with torch.no_grad():
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predicted_probs.append(predict.numpy())
                predicted_class = torch.argmax(predict).item()
                # if torch.max(predict).item() >0.7:
                #
                #     predicted_class = torch.argmax(predict).item()
                # else:
                #     predicted_class = torch.argmin(predict).item()
                predicted_classes.append(predicted_class)
                true_labels.append(0)  # 'n' label
                
                # Write to file
                f.write(f"{img_path}, 0, {predicted_class}, {predict.numpy()}\n")
        
        # Predict for 'y' class (label 1)
        for img_path in tqdm(y_images, desc="Processing class 'y' images"):
            img = process_image(img_path, data_transform)
            with torch.no_grad():
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predicted_probs.append(predict.numpy())
                predicted_class = torch.argmax(predict).item()
                #
                # if torch.max(predict).item() > 0.8:
                #
                #     predicted_class = torch.argmax(predict).item()
                # else:
                #     predicted_class = torch.argmin(predict).item()
                predicted_classes.append(predicted_class)
                true_labels.append(1)  # 'y' label
                
                # Write to file
                f.write(f"{img_path}, 1, {predicted_class}, {predict.numpy()}\n")

    # Convert lists to arrays for easier handling
    predicted_probs = np.array(predicted_probs)
    predicted_classes = np.array(predicted_classes)
    true_labels = np.array(true_labels)

    # Separate probabilities based on true labels
    # n_probs = predicted_probs[true_labels == 0]  # True label 'n'
    # y_probs = predicted_probs[true_labels == 1]  # True label 'y'

    # Plot probability distribution for true label 'n'
    # plt.figure(figsize=(10, 6))
    # plt.hist(n_probs[:, 0], bins=50, alpha=0.7, label='Predicted Class n Probability', color='blue')
    # plt.hist(n_probs[:, 1], bins=50, alpha=0.5, label='Predicted Class y Probability', color='orange')
    # plt.xlabel('Probability')
    # plt.ylabel('Frequency')
    # plt.title('Probability Distribution for True Class n')
    # plt.legend()
    # plt.show()
    #
    # # Plot probability distribution for true label 'y'
    # plt.figure(figsize=(10, 6))
    # plt.hist(y_probs[:, 0], bins=50, alpha=0.7, label='Predicted Class n Probability', color='blue')
    # plt.hist(y_probs[:, 1], bins=50, alpha=0.5, label='Predicted Class y Probability', color='orange')
    # plt.xlabel('Probability')
    # plt.ylabel('Frequency')
    # plt.title('Probability Distribution for True Class y')
    # plt.legend()
    # plt.show()

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['n', 'y'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == '__main__':
    main()

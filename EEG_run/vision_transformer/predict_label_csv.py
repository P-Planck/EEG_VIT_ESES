import os
import json
import csv

import torch
from PIL import Image
from torchvision import transforms
from vit_model import vit_base_patch16_224_in21k as create_model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Transform for the input images
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # Load the class indices mapping
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"File: '{json_path}' does not exist."
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # Create and load the model
    model = create_model(num_classes=2, has_logits=False).to(device)
    model_weight_path = "./weights/model-0.pth"
    assert os.path.exists(model_weight_path), f"Model weights not found at '{model_weight_path}'."
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # Define folder paths
    folders = {"n": 0, "y": 1}
    root_dir = "EEG_data/EEG_IMG"
    results = []

    # Iterate over each folder and process images
    for folder, actual_label in folders.items():
        folder_path = os.path.join(root_dir, folder)
        assert os.path.exists(folder_path), f"Folder: '{folder_path}' does not exist."

        for img_name in os.listdir(folder_path):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(folder_path, img_name)
                img = Image.open(img_path)
                img = data_transform(img)
                img = torch.unsqueeze(img, dim=0).to(device)

                # Predict the label
                with torch.no_grad():
                    output = torch.squeeze(model(img)).cpu()
                    predict = torch.softmax(output, dim=0)
                    predict_cla = torch.argmax(predict).item()

                # Determine the result value
                if predict_cla == actual_label:
                    result_value = 0  # Correct prediction
                elif actual_label == 1 and predict_cla == 0:
                    result_value = 1  # `y` predicted as `n`
                elif actual_label == 0 and predict_cla == 1:
                    result_value = 2  # `n` predicted as `y`

                # Store the result
                results.append({"image": img_name, "result": result_value})

    # Write results to a CSV file
    csv_path = "prediction_results.csv"
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["image", "result"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Prediction results saved to '{csv_path}'.")

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import model
from models.model import get_model

# Cấu hình thiết bị
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(data_dir, model_path, batch_size=32):
    print(f"Evaluating on: {DEVICE}")
    
    # 1. Prepare Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Use 'val' or 'test' directory
    eval_dir = os.path.join(data_dir, 'val')
    if not os.path.exists(eval_dir):
        # Fallback to test if val doesn't exist (or user named it test)
        eval_dir = os.path.join(data_dir, 'test')
    
    if not os.path.exists(eval_dir):
        print(f"Error: Validation/Test directory not found at {eval_dir}")
        return

    dataset = datasets.ImageFolder(eval_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    class_names = dataset.classes
    print(f"Classes: {class_names}")

    # 2. Load Model
    model = get_model(num_classes=2, feature_extract=True)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Error: Model path '{model_path}' not found.")
        return

    model.to(DEVICE)
    model.eval()

    # 3. Evaluation Loop
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save plot
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/confusion_matrix.png")
    print("Confusion Matrix saved to outputs/confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/raw', help='Path to data directory')
    parser.add_argument('--model_path', type=str, default='../outputs/pneumonia_model.pth', help='Path to trained model')
    args = parser.parse_args()
    
    evaluate_model(args.data_dir, args.model_path)

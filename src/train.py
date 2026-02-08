import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import argparse
from tqdm import tqdm
import copy

# Import model
from models.model import get_model
from utils import plot_training_history
from data.loader import get_dataloaders

# Cấu hình thiết bị (GPU/CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(data_dir, num_epochs=10, batch_size=32, learning_rate=0.001, output_dir="outputs"):
    print(f"Training on: {DEVICE}")
    
    # 1. & 2. Chuẩn bị dữ liệu và Load Dataset
    try:
        dataloaders, dataset_sizes, class_names = get_dataloaders(data_dir, batch_size=batch_size)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # 3. Khởi tạo Model, Loss, Optimizer
    model = get_model(num_classes=2, feature_extract=True) # Transfer Learning
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Track metrics
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # 4. Training Loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            # Use tqdm only for training phase to avoid clutter
            iterator = tqdm(dataloaders[phase], desc=phase) if phase == 'train' else dataloaders[phase]
            
            for inputs, labels in iterator:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # Deep copy the model if it's the best one so far
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

    print(f"Training Complete! Best Val Acc: {best_acc:.4f}")
    
    # 5. Save Best Model
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, "pneumonia_model.pth")
    torch.save(best_model_wts, model_save_path)
    print(f"Best model saved to {model_save_path}")
    
    # Plot history
    plot_training_history(history, save_path=os.path.join(output_dir, "training_history.png"))

if __name__ == "__main__":
    # Xác định đường dẫn gốc của project
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    default_data_dir = os.path.join(project_root, 'data', 'raw')

    parser = argparse.ArgumentParser(description="Train Pneumonia Detection Model")
    parser.add_argument('--data_dir', type=str, default=default_data_dir, help='Path to data directory containing train/val folders')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save outputs')
    
    args = parser.parse_args()
    
    # Kiểm tra đường dẫn data trước khi chạy
    if os.path.exists(os.path.join(args.data_dir, 'train')):
        train_model(args.data_dir, num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, output_dir=args.output_dir)
    else:
        print(f"Error: Data directory '{args.data_dir}' not found or structure incorrect.")
        print("Please download dataset and ensure 'train' and 'val' folders exist.")


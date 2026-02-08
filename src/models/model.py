import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=2, feature_extract=True):
    """
    Khởi tạo mô hình ResNet18 Pre-trained.
    
    Args:
        num_classes (int): Số lượng lớp output (2 cho Normal vs Pneumonia).
        feature_extract (bool): Nếu True, đóng băng các lớp feature extraction, chỉ train classifier.
        
    Returns:
        model: Mô hình PyTorch.
    """
    # 1. Tải model pre-trained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # 2. Đóng băng trọng số (nếu feature_extract=True)
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
            
    # 3. Thay thế lớp Fully Connected cuối cùng (Classifier)
    # ResNet18: fc input features = 512
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
        # Không cần Softmax nếu dùng CrossEntropyLoss
    )
    
    return model

if __name__ == "__main__":
    # Test model structure
    net = get_model()
    print(net)

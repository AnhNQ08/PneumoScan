import torch
from torchvision import transforms
from PIL import Image
import argparse
import os

# Import model
from models.model import get_model

# Cấu hình thiết bị
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_single_image(image_path, model_path, show_image=False):
    """
    Dự đoán nhãn cho một ảnh input.
    """
    # 1. Load Model
    model = get_model(num_classes=2, feature_extract=True)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print(f"Error: Model path '{model_path}' not found.")
        return None, None

    model.to(DEVICE)
    model.eval()

    # 2. Xử lý ảnh (Transform)
    # Phải giống params trong train.py (Validation transform)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0) # Add batch dimension e.g. (1, 3, 224, 224)
        input_tensor = input_tensor.to(DEVICE)

        # 3. Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

        class_names = ['NORMAL', 'PNEUMONIA'] # Cần khớp với ImageFolder classes index
        prediction = class_names[preds.item()]
        confidence = probabilities[0][preds.item()].item()

        return prediction, confidence

    except Exception as e:
        print(f"Error predicting image: {e}")
        return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Pneumonia from X-Ray Image")
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--model_path', type=str, default='outputs/pneumonia_model.pth', help='Path to trained model')
    args = parser.parse_args()

    # Kiểm tra đường dẫn ảnh
    if not os.path.exists(args.image_path):
        print(f"Error: Image path '{args.image_path}' not found.")
        exit(1)
        
    # Sửa đường dẫn model mặc định nếu chạy từ root
    if args.model_path == 'outputs/pneumonia_model.pth' and not os.path.exists(args.model_path):
        # Thử tìm ở ../outputs nếu đang chạy trong src
        alt_path = '../outputs/pneumonia_model.pth'
        if os.path.exists(alt_path):
            args.model_path = alt_path

    prediction, confidence = predict_single_image(args.image_path, args.model_path)
    
    if prediction:
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence*100:.2f}%")


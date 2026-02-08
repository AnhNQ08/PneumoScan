import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from models.model import get_model
import os
import matplotlib.pyplot as plt

# Cấu hình
st.set_page_config(page_title="PneumoScan", page_icon="🫁")

# Load model
@st.cache_resource
def load_model(model_path="outputs/pneumonia_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=2, feature_extract=True)
    
    # Try finding the model relative to different potential current working directories
    if not os.path.exists(model_path):
        # Maybe running from src/
        alt_path = "../outputs/pneumonia_model.pth"
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            return None, "Model file not found. Please train the model first."
            
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)

# Preprocessing
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# UI
st.title("🫁 PneumoScan: Pneumonia Detection")
st.markdown("Upload a chest X-Ray image to detect if it shows signs of Pneumonia.")

uploaded_file = st.file_uploader("Choose an X-Ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    with col2:
        st.write("Analyzing...")
        
        # Load model
        model, error = load_model()
        
        if error:
            st.error(f"Error loading model: {error}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            input_tensor = process_image(image).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

            class_names = ['NORMAL', 'PNEUMONIA']
            prediction = class_names[preds.item()]
            confidence = probabilities[0][preds.item()].item()

            if prediction == "PNEUMONIA":
                st.error(f"**Prediction: {prediction}**")
            else:
                st.success(f"**Prediction: {prediction}**")
                
            st.metric(label="Confidence", value=f"{confidence*100:.2f}%")
            
            # Show probability bar
            st.write("Probability Distribution:")
            prob_dict = {class_names[i]: float(probabilities[0][i]) for i in range(2)}
            st.bar_chart(prob_dict)

st.markdown("---")
st.info("Note: This tool is for educational purposes only and not for medical diagnosis.")

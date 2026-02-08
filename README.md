# PneumoScan: Pneumonia Detection from Chest X-Ray Images using Deep Learning

![Pneumonia Detection Banner](https://i.imgur.com/j5w9y7s.png) 
*(Note: Replace with your own banner or example image)*

## 1. Project Overview
This project leverages Deep Learning, specifically Convolutional Neural Networks (CNNs), to detect Pneumonia from chest X-Ray images. We utilize **Transfer Learning** with a pre-trained **ResNet18** architecture to achieve high accuracy even with limited computational resources.

The problem is formulated as a binary classification task:
- **NORMAL**: Healthy lungs.
- **PNEUMONIA**: Infected lungs.

This project demonstrates an end-to-end Machine Learning pipeline:
1.  **Data Loading & Augmentation**: Handling medical imaging data.
2.  **Model Training**: Fine-tuning a pre-trained ResNet18 model using PyTorch.
3.  **Evaluation**: rigorous testing with Confusion Matrix, Accuracy, Precision, Recall, and F1-Score.
4.  **Inference**: A script to predict pneumonia on new, unseen images.

## 2. Dataset
The dataset used is the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) from Kaggle.
- **Total Images**: ~5,800 JPEG images.
- **Categories**: Normal, Pneumonia.
- **Data Split**: The dataset is organized into 3 folders (train, test, val). We treat the 'test' set as our validation set during training for robust evaluation.

### Directory Structure
Ensure your project directory looks like this after downloading data:
```text
PneumoScan/
├── data/
│   └── raw/
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       └── val/  <-- (Rename 'test' folder from Kaggle to 'val')
│           ├── NORMAL/
│           └── PNEUMONIA/
├── notebooks/          # Jupyter Notebooks for EDA
├── outputs/            # Saved models and plots
├── src/
│   ├── data/           # Data loading scripts
│   ├── models/         # Model architecture (ResNet18)
│   ├── train.py        # Training script
│   ├── evaluate.py     # Evaluation script
│   ├── predict.py      # Inference script
│   └── utils.py        # Helper functions
├── requirements.txt
└── README.md
```

3. Installation
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/AnhNQ08/PneumoScan.git
    cd PneumoScan
    ```

2.  **Install dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## 4. Usage

### 4.1 Training
To train the model, run the following command. The script will automatically download the pre-trained ResNet18 weights.
```bash
python src/train.py --epochs 10 --batch_size 32
```
- **Features**: Automatically saves the best model based on validation accuracy to `outputs/pneumonia_model.pth`.
- **Logging**: Displays loss and accuracy per epoch.

### 4.2 Evaluation
To evaluate the trained model on the validation/test set and generate a **Classification Report** and **Confusion Matrix**:
```bash
python src/evaluate.py --data_dir data/raw --model_path outputs/pneumonia_model.pth
```
This will save the confusion matrix plot to `outputs/confusion_matrix.png`.

### 4.3 Prediction (Inference)
To predict the class of a single image:
```bash
python src/predict.py --image_path path/to/your/image.jpeg
```
**Output Example**:
```text
Prediction: PNEUMONIA
Confidence: 98.45%
```

### 4.4 Web App (Demo)
Launch a simple web interface to upload images and get predictions:
```bash
streamlit run src/app.py
```
This will open a local web server (e.g., `http://localhost:8501`) where you can interact with the model.

## 5. Results
*(You can update this section with your actual training results after running the code)*

Example Metrics:
- **Accuracy**: > 90%
- **F1-Score**: > 0.90

## 6. Technologies Used
- **Language**: Python 3.8+
- **Framework**: PyTorch
- **Web App**: Streamlit
- **Libraries**: Torchvision, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, PIL.

## 7. Future Improvements
- Implement Grad-CAM to visualize which parts of the X-Ray the model focuses on.
- Experiment with deeper architectures like ResNet50 or DenseNet121.
- Deploy as a web app using Streamlit or Flask.

## 8. Author
**AnhNQ08**
- [GitHub Profile](https://github.com/AnhNQ08)
- [LinkedIn](https://linkedin.com/in/yourprofile)


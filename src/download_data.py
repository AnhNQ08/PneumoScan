import os
import shutil
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_and_setup_data():
    print("Authenticating with Kaggle...")
    api = KaggleApi()
    api.authenticate()

    dataset = 'paultimothymooney/chest-xray-pneumonia'
    download_path = 'data/raw'
    
    # Ensure directory exists
    os.makedirs(download_path, exist_ok=True)
    
    print(f"Downloading {dataset} to {download_path}...")
    # Download and unzip
    api.dataset_download_files(dataset, path=download_path, unzip=True)
    print("Download complete.")
    
    # Organize folders
    # Kaggle structure: data/raw/chest_xray/train, test, val
    # Target structure: data/raw/train, data/raw/val (using test as val)
    
    source_root = os.path.join(download_path, 'chest_xray')
    if not os.path.exists(source_root):
        print("Error: 'chest_xray' folder not found after unzip. Check download path.")
        return

    # 1. Setup Train
    print("Setting up Train data...")
    source_train = os.path.join(source_root, 'train')
    target_train = os.path.join(download_path, 'train')
    
    if os.path.exists(target_train):
        shutil.rmtree(target_train) # Clean existing to avoid duplicates
    shutil.move(source_train, target_train)
    
    # 2. Setup Val (Using Test set from Kaggle as Val)
    print("Setting up Validation data (using Kaggle 'test' set)...")
    source_test = os.path.join(source_root, 'test')
    target_val = os.path.join(download_path, 'val')
    
    if os.path.exists(target_val):
        shutil.rmtree(target_val)
    shutil.move(source_test, target_val)
    
    # 3. Cleanup
    print("Cleaning up temporary files...")
    shutil.rmtree(source_root) # Remove chest_xray folder and original 'val' folder inside it
    # Also remove the duplicate chest_xray folder if it exists (sometimes Kaggle unzips weirdly)
    mac_idx = os.path.join(download_path, '__MACOSX')
    if os.path.exists(mac_idx):
        shutil.rmtree(mac_idx)

    print("Data setup complete!")
    print(f"Train path: {target_train}")
    print(f"Val path: {target_val}")

if __name__ == "__main__":
    download_and_setup_data()

from PIL import Image
import os
import numpy as np

def create_dummy_image(path):
    # Tạo ảnh ngẫu nhiên 224x224
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(path)

dirs = [
    "data/raw/train/NORMAL", "data/raw/train/PNEUMONIA",
    "data/raw/val/NORMAL", "data/raw/val/PNEUMONIA"
]

for d in dirs:
    os.makedirs(d, exist_ok=True) # Tạo thư mục nếu chưa có
    prefix = d.split('/')[-1]
    for i in range(5): # Tạo 5 ảnh mỗi loại
        create_dummy_image(f"{d}/{prefix}_{i}.jpg")
    print(f"Created dummy images in {d}")

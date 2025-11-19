import numpy as np
import matplotlib.pyplot as plt
from skimage import data, feature
from skimage.color import rgb2gray
from PIL import Image
import tkinter as tk
from tkinter import filedialog

def load_and_process_image():
    # Tải ảnh từ file
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    if file_path:
        # Đọc ảnh
        image = Image.open(file_path)
        image_array = np.array(image)
        
        # Chuyển sang grayscale nếu cần
        if len(image_array.shape) == 3:
            gray_image = rgb2gray(image_array)
        else:
            gray_image = image_array
            
        return image_array, gray_image
    return None, None

def compute_hog(image):
    # Tính HOG features
    hog_features, hog_image = feature.hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        block_norm='L2-Hys'
    )
    return hog_features, hog_image

# Tạo GUI đơn giản
root = tk.Tk()
root.withdraw()  # Ẩn cửa sổ chính

# Load và xử lý ảnh
original_image, gray_image = load_and_process_image()

if original_image is not None:
    # Tính HOG
    hog_features, hog_image = compute_hog(gray_image)
    
    # Hiển thị kết quả
    plt.figure(figsize=(12, 6))
    
    # Ảnh gốc
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    if len(original_image.shape) == 3:
        plt.imshow(original_image)
    else:
        plt.imshow(original_image, cmap='gray')
    plt.axis('off')
    
    # HOG features
    plt.subplot(1, 2, 2)
    plt.title('HOG Features')
    plt.imshow(hog_image, cmap='hot')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"HOG features shape: {hog_features.shape}")
    print(f"Number of HOG features: {len(hog_features)}")
else:
    print("Không có ảnh được chọn!")

root.destroy()
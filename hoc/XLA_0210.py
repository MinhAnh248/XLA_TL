import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Đọc ảnh gốc
image_path = 'chupanh.jpg'
image = Image.open(image_path).convert('L')
image_array = np.array(image, dtype=np.float32)

# Hàm tạo nhiễu muối (salt)
def add_salt_noise(image, prob=0.5):
    noisy = image.copy()
    salt_mask = np.random.random(image.shape) < prob
    noisy[salt_mask] = 255
    return noisy

# Hàm tạo nhiễu tiêu (pepper)
def add_pepper_noise(image, prob=1):
    noisy = image.copy()
    pepper_mask = np.random.random(image.shape) < prob
    noisy[pepper_mask] = 0
    return noisy

# Hàm tạo nhiễu muối tiêu
def add_salt_pepper_noise(image, salt_prob=0.025, pepper_prob=0.025):
    noisy = image.copy()
    salt_mask = np.random.random(image.shape) < salt_prob
    pepper_mask = np.random.random(image.shape) < pepper_prob
    noisy[salt_mask] = 255
    noisy[pepper_mask] = 0
    return noisy

# Tạo các ảnh nhiễu
salt_image = add_salt_noise(image_array, prob=0.1)
pepper_image = add_pepper_noise(image_array, prob=0.1)
salt_pepper_image = add_salt_pepper_noise(image_array, salt_prob=0.075, pepper_prob=0.025)

# Hiển thị kết quả
plt.figure(figsize=(20, 5))

# Ảnh gốc
plt.subplot(1, 4, 1)
plt.title('Original Image')
plt.imshow(image_array, cmap='gray')
plt.axis('off')

# Ảnh nhiễu muối
plt.subplot(1, 4, 2)
plt.title('Salt Noise')
plt.imshow(salt_image, cmap='gray')
plt.axis('off')

# Ảnh nhiễu tiêu
plt.subplot(1, 4, 3)
plt.title('Pepper Noise')
plt.imshow(pepper_image, cmap='gray')
plt.axis('off')

# Ảnh nhiễu muối tiêu
plt.subplot(1, 4, 4)
plt.title('Salt & Pepper Noise')
plt.imshow(salt_pepper_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
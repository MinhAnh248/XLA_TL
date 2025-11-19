import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift
from PIL import Image

# Đọc ảnh
image_path = 'cogai.jpg'
image = Image.open(image_path).convert('L')
image_array = np.array(image)

# Step 1: Biến đổi Fourier rời rạc bằng fft2
G_uv = fft2(image_array)

# Step 2: Tính ma trận độ lớn của số phức
magnitude = np.abs(G_uv)

# Step 3: Chuyển trục tọa độ 0,0 vào giữa
magnitude_shifted = fftshift(magnitude)

# Step 4: Dùng hàm logarit để biểu diễn biên độ
log_magnitude = np.log1p(magnitude_shifted)
# Hiển thị hình ảnh gốc và phổ theo thang logarit
plt.figure(figsize=(12, 6))

# Ảnh gốc
plt.subplot(3, 2, 1)
plt.title('Image')
plt.imshow(image_array, cmap='gray')
plt.axis('off')

# Phổ ảnh theo thang logarit
plt.subplot(3, 2, 2)
plt.title('Log Magnitude Spectrum with center')
plt.imshow(log_magnitude, cmap='hot')
plt.axis('off')

plt.subplot(3, 2, 3)
plt.title('Log Magnitude Spectrum with center')
plt.imshow(log_magnitude, cmap='hot')
plt.axis('off')

plt.subplot(3, 2, 4)
plt.title('Image')
plt.imshow(image_array, cmap='gray')
plt.axis('off')

plt.subplot(3, 2, 5)
plt.title('Image')
plt.imshow(image_array, cmap='gray')
plt.axis('off')

plt.subplot(3, 2, 6)
plt.title('Log Magnitude Spectrum with center')
plt.imshow(log_magnitude, cmap='hot')
plt.axis('off')



plt.show()
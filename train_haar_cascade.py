import cv2
import os
import numpy as np

def create_positive_samples():
    """Tạo positive samples từ dataset"""
    dataset_path = "dataset"
    positive_path = "positive_images"
    os.makedirs(positive_path, exist_ok=True)
    
    sample_count = 0
    
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if os.path.isdir(person_path):
            for filename in os.listdir(person_path):
                if filename.endswith('.jpg'):
                    src_path = os.path.join(person_path, filename)
                    dst_path = os.path.join(positive_path, f"pos_{sample_count:04d}.jpg")
                    
                    # Copy và resize ảnh
                    img = cv2.imread(src_path)
                    if img is not None:
                        img_resized = cv2.resize(img, (24, 24))
                        cv2.imwrite(dst_path, img_resized)
                        sample_count += 1
    
    print(f"Created {sample_count} positive samples")
    return sample_count

def create_negative_samples():
    """Tạo negative samples (ảnh không có khuôn mặt)"""
    negative_path = "negative_images"
    os.makedirs(negative_path, exist_ok=True)
    
    # Tạo ảnh ngẫu nhiên làm negative samples
    for i in range(1000):
        # Tạo ảnh noise ngẫu nhiên
        noise = np.random.randint(0, 255, (24, 24, 3), dtype=np.uint8)
        cv2.imwrite(f"{negative_path}/neg_{i:04d}.jpg", noise)
    
    print("Created 1000 negative samples")

def create_description_files(pos_count):
    """Tạo file mô tả cho positive và negative samples"""
    
    # Tạo positive description file
    with open("positive_images.txt", "w") as f:
        for i in range(pos_count):
            f.write(f"positive_images/pos_{i:04d}.jpg 1 0 0 24 24\n")
    
    # Tạo negative description file
    with open("negative_images.txt", "w") as f:
        for i in range(1000):
            f.write(f"negative_images/neg_{i:04d}.jpg\n")
    
    print("Created description files")

def train_cascade():
    """Train Haar Cascade"""
    print("Training Haar Cascade...")
    print("This process can take several hours...")
    
    # Tạo thư mục output
    os.makedirs("cascade_output", exist_ok=True)
    
    # Lệnh train (cần OpenCV tools)
    train_command = f"""
    opencv_createsamples -info positive_images.txt -bg negative_images.txt -vec samples.vec -w 24 -h 24
    opencv_traincascade -data cascade_output -vec samples.vec -bg negative_images.txt -numPos 800 -numNeg 1000 -w 24 -h 24 -mode ALL -precalcValBufSize 2048 -precalcIdxBufSize 2048
    """
    
    print("Run these commands manually:")
    print(train_command)

if __name__ == "__main__":
    print("Haar Cascade Training Process")
    print("1. Creating positive samples...")
    pos_count = create_positive_samples()
    
    print("2. Creating negative samples...")
    create_negative_samples()
    
    print("3. Creating description files...")
    create_description_files(pos_count)
    
    print("4. Training cascade...")
    train_cascade()
    
    print("\nNote: You need OpenCV development tools to complete training.")
    print("Install: pip install opencv-contrib-python")
    print("Or use online training services.")
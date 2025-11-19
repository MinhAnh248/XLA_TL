import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk

class FaceDetector:
    def __init__(self):
        # Load Haar cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
    def detect_features(self, image):
        """Nhận diện khuôn mặt, mắt và nụ cười"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Tạo ảnh kết quả
        result_image = image.copy()
        
        for (x, y, w, h) in faces:
            # Vẽ khung mặt (màu xanh lá)
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(result_image, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # ROI cho mặt
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = result_image[y:y+h, x:x+w]
            
            # Detect eyes trong vùng mặt
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
                cv2.putText(roi_color, 'Eye', (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Detect smile trong vùng mặt
            smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
                cv2.putText(roi_color, 'Smile', (sx, sy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return result_image, len(faces), len(eyes) if 'eyes' in locals() else 0, len(smiles) if 'smiles' in locals() else 0

def load_image():
    """Tải ảnh từ file"""
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh để nhận diện khuôn mặt",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    root.destroy()
    
    if file_path:
        image = cv2.imread(file_path)
        return image
    return None

def main():
    # Khởi tạo detector
    detector = FaceDetector()
    
    # Load ảnh
    image = load_image()
    
    if image is None:
        print("Không có ảnh được chọn!")
        return
    
    # Nhận diện
    result_image, num_faces, num_eyes, num_smiles = detector.detect_features(image)
    
    # Chuyển đổi màu cho matplotlib (BGR -> RGB)
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    # Hiển thị kết quả
    plt.figure(figsize=(15, 8))
    
    # Ảnh gốc
    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_rgb)
    plt.axis('off')
    
    # Ảnh với detection
    plt.subplot(2, 2, 2)
    plt.title('Face Detection Result')
    plt.imshow(result_rgb)
    plt.axis('off')
    
    # Thống kê
    plt.subplot(2, 2, 3)
    features = ['Faces', 'Eyes', 'Smiles']
    counts = [num_faces, num_eyes, num_smiles]
    colors = ['green', 'blue', 'red']
    
    bars = plt.bar(features, counts, color=colors, alpha=0.7)
    plt.title('Detection Statistics')
    plt.ylabel('Count')
    
    # Thêm số lượng trên mỗi cột
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # Thông tin chi tiết
    plt.subplot(2, 2, 4)
    plt.axis('off')
    info_text = f"""
    HAAR CASCADE DETECTION RESULTS
    
    📊 Statistics:
    • Faces detected: {num_faces}
    • Eyes detected: {num_eyes}
    • Smiles detected: {num_smiles}
    
    🎯 Detection Info:
    • Face cascade: haarcascade_frontalface_default.xml
    • Eye cascade: haarcascade_eye.xml
    • Smile cascade: haarcascade_smile.xml
    
    🎨 Color coding:
    • Green rectangles: Faces
    • Blue rectangles: Eyes
    • Red rectangles: Smiles
    """
    
    plt.text(0.1, 0.9, info_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n✅ Detection completed!")
    print(f"📊 Found: {num_faces} faces, {num_eyes} eyes, {num_smiles} smiles")

if __name__ == "__main__":
    main()
import cv2
import os
import numpy as np
from datetime import datetime

class FaceDetection:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        self.dataset_path = "dataset"
        self.current_person = "ThuyTien"
        self.create_directories()
        
    def create_directories(self):
        """Tạo thư mục dataset nếu chưa tồn tại"""
        os.makedirs(self.dataset_path, exist_ok=True)
        
    def set_current_person(self, person_name):
        """Đặt tên người hiện tại"""
        self.current_person = person_name
        person_path = os.path.join(self.dataset_path, person_name)
        os.makedirs(person_path, exist_ok=True)
        
    def detect_faces(self, frame):
        """Phát hiện khuôn mặt trong frame với tham số tối ưu"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Cải thiện chất lượng ảnh
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,   # Tăng lên để nhanh hơn
            minNeighbors=5,    # Giảm xuống để nhanh hơn
            minSize=(30, 30),  # Giảm kích thước
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces
    
    def save_face_samples(self, frame, faces):
        """Lưu mẫu khuôn mặt vào dataset/[person_name]"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        saved_files = []
        person_path = os.path.join(self.dataset_path, self.current_person)
        
        for i, (x, y, w, h) in enumerate(faces):
            # Cắt vùng khuôn mặt
            face_roi = frame[y:y+h, x:x+w]
            
            # Resize về kích thước chuẩn
            face_resized = cv2.resize(face_roi, (160, 160))
            
            # Tên file
            filename = f"{self.current_person}_{timestamp}_{i}.jpg"
            filepath = os.path.join(person_path, filename)
            
            # Chỉ lưu vào dataset/[person_name]
            cv2.imwrite(filepath, face_resized)
            
            saved_files.append(filepath)
            
        return saved_files
    
    def detect_facial_features(self, frame, face_roi, x, y):
        """Phát hiện mắt và môi trong khuôn mặt"""
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Phát hiện mắt
        eyes = self.eye_cascade.detectMultiScale(gray_roi, 1.1, 3, minSize=(10, 10))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 0), 1)
            cv2.putText(frame, 'Eye', (x+ex, y+ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        # Phát hiện môi
        mouths = self.mouth_cascade.detectMultiScale(gray_roi, 1.1, 3, minSize=(10, 10))
        for (mx, my, mw, mh) in mouths:
            cv2.rectangle(frame, (x+mx, y+my), (x+mx+mw, y+my+mh), (255, 255, 0), 1)
            cv2.putText(frame, 'Mouth', (x+mx, y+my-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
    
    def draw_faces(self, frame, faces):
        """Vẽ khung bao quanh khuôn mặt và các bộ phận"""
        for i, (x, y, w, h) in enumerate(faces):
            # Vẽ khung khuôn mặt
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'Face {i+1}', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Phát hiện các bộ phận khuôn mặt
            face_roi = frame[y:y+h, x:x+w]
            self.detect_facial_features(frame, face_roi, x, y)
        
        return frame
    
    def get_sample_count(self):
        """Đếm số mẫu đã lưu cho người hiện tại"""
        person_path = os.path.join(self.dataset_path, self.current_person)
        if os.path.exists(person_path):
            return len([f for f in os.listdir(person_path) if f.endswith('.jpg')])
        return 0
        
    def get_all_persons(self):
        """Lấy danh sách tất cả người trong dataset"""
        persons = []
        if os.path.exists(self.dataset_path):
            for item in os.listdir(self.dataset_path):
                item_path = os.path.join(self.dataset_path, item)
                if os.path.isdir(item_path):
                    persons.append(item)
        return persons
    
    def get_recent_samples(self, limit=10):
        """Lấy danh sách mẫu gần nhất từ dataset"""
        samples = []
        person_path = os.path.join(self.dataset_path, self.current_person)
        if os.path.exists(person_path):
            files = sorted(os.listdir(person_path), reverse=True)
            for filename in files[:limit]:
                if filename.endswith('.jpg'):
                    # Tạo đường dẫn tương đối
                    samples.append(f'/dataset/{self.current_person}/{filename}')
        return samples
    
    def clear_samples(self):
        """Xóa tất cả mẫu của người hiện tại"""
        person_path = os.path.join(self.dataset_path, self.current_person)
        if os.path.exists(person_path):
            for filename in os.listdir(person_path):
                if filename.endswith('.jpg'):
                    os.remove(os.path.join(person_path, filename))
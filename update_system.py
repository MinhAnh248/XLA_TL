import shutil
import os

def update_face_recognizer():
    """Cập nhật face_recognizer.py để sử dụng LBPH"""
    
    new_code = '''import cv2
import numpy as np
import os

class FaceRecognizer:
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.dataset_path = 'dataset'
        self.model_path = 'trained_model.yml'
        self.trained = False
        self.name_to_id = {'ThuyTien': 1, 'MinhAnh': 2, 'ThinhNho': 3}
        self.id_to_name = {1: 'ThuyTien', 2: 'MinhAnh', 3: 'ThinhNho'}
        
    def train_model(self):
        """Train LBPH model"""
        faces, ids = self.get_images_and_labels()
        if len(faces) > 0:
            self.recognizer.train(faces, np.array(ids))
            self.recognizer.write(self.model_path)
            self.trained = True
            return True
        return False
    
    def load_model(self):
        """Load trained model"""
        if os.path.exists(self.model_path):
            self.recognizer.read(self.model_path)
            self.trained = True
            return True
        return False
    
    def get_images_and_labels(self):
        """Lấy ảnh và labels từ dataset"""
        faces = []
        ids = []
        
        for person_name, person_id in self.name_to_id.items():
            person_path = os.path.join(self.dataset_path, person_name)
            if os.path.exists(person_path):
                for filename in os.listdir(person_path):
                    if filename.endswith('.jpg'):
                        img_path = os.path.join(person_path, filename)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            faces.append(img)
                            ids.append(person_id)
        
        return faces, ids
    
    def recognize_face(self, face_img):
        """Nhận diện khuôn mặt bằng LBPH"""
        if not self.trained:
            return "Unknown", "0%"
        
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
        
        id_, confidence = self.recognizer.predict(gray)
        
        if confidence < 50:  # Ngưỡng tin cậy
            name = self.id_to_name.get(id_, "Unknown")
            confidence_text = f"{100-confidence:.1f}%"
            return name, confidence_text
        else:
            return "Unknown", f"{100-confidence:.1f}%"
'''
    
    # Backup file cũ
    if os.path.exists('face_recognizer.py'):
        shutil.copy('face_recognizer.py', 'face_recognizer_backup.py')
    
    # Ghi file mới
    with open('face_recognizer.py', 'w', encoding='utf-8') as f:
        f.write(new_code)
    
    print("Updated face_recognizer.py to use LBPH")

if __name__ == "__main__":
    update_face_recognizer()
    print("System updated successfully!")
    print("Now run: python app.py")
import cv2
import numpy as np
import os
from PIL import Image

class SimpleFaceTrainer:
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def get_images_and_labels(self, path):
        """Lấy ảnh và labels từ dataset"""
        image_paths = []
        for person_folder in os.listdir(path):
            person_path = os.path.join(path, person_folder)
            if os.path.isdir(person_path):
                for filename in os.listdir(person_path):
                    if filename.endswith('.jpg'):
                        image_paths.append(os.path.join(person_path, filename))
        
        face_samples = []
        ids = []
        
        # Mapping tên người thành ID
        name_to_id = {'ThuyTien': 1, 'MinhAnh': 2, 'ThinhNho': 3}
        
        for image_path in image_paths:
            # Lấy tên người từ đường dẫn
            person_name = os.path.basename(os.path.dirname(image_path))
            person_id = name_to_id.get(person_name, 0)
            
            # Đọc ảnh
            img = Image.open(image_path).convert('L')
            img_np = np.array(img, 'uint8')
            
            face_samples.append(img_np)
            ids.append(person_id)
            
        return face_samples, ids
    
    def train_model(self):
        """Train model với dữ liệu từ dataset"""
        print("Training model...")
        
        faces, ids = self.get_images_and_labels('dataset')
        
        if len(faces) == 0:
            print("No training data found!")
            return False
        
        # Train model
        self.recognizer.train(faces, np.array(ids))
        
        # Lưu model
        self.recognizer.write('trained_model.yml')
        print(f"Model trained with {len(faces)} samples")
        return True
    
    def test_model(self):
        """Test model với camera"""
        if not os.path.exists('trained_model.yml'):
            print("No trained model found!")
            return
        
        # Load model
        self.recognizer.read('trained_model.yml')
        
        # Mapping ID thành tên
        id_to_name = {1: 'ThuyTien', 2: 'MinhAnh', 3: 'ThinhNho'}
        
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                
                # Nhận diện
                id_, confidence = self.recognizer.predict(face_roi)
                
                if confidence < 50:  # Ngưỡng tin cậy
                    name = id_to_name.get(id_, "Unknown")
                    confidence_text = f"{100-confidence:.1f}%"
                    color = (0, 255, 0)
                else:
                    name = "Unknown"
                    confidence_text = f"{100-confidence:.1f}%"
                    color = (0, 0, 255)
                
                # Vẽ khung và tên
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f'{name}', (x, y-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f'{confidence_text}', (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.imshow('Face Recognition Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    trainer = SimpleFaceTrainer()
    
    print("1. Train model")
    print("2. Test model")
    choice = input("Choose option (1/2): ")
    
    if choice == '1':
        trainer.train_model()
    elif choice == '2':
        trainer.test_model()
    else:
        print("Invalid choice")
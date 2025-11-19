import cv2
import numpy as np
import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

class CameraFaceDetector:
    def __init__(self):
        # Load 3 Haar cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Khởi tạo camera
        self.cap = cv2.VideoCapture(0)
        
        # Dataset settings
        self.save_dataset = False
        self.dataset_counter = {}
        self.selected_person = "DuongBao"  # Default
        
        # Face recognition
        self.face_recognizer = None
        self.label_encoder = None
        self.recognition_enabled = False
        
        self.create_dataset_folders()
        self.setup_gui()
        self.load_or_train_model()
        
    def detect_features(self, frame):
        """Nhận diện khuôn mặt, mắt và nụ cười trong frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        total_eyes = 0
        total_smiles = 0
        
        for i, (x, y, w, h) in enumerate(faces):
            # Vẽ khung mặt (màu xanh lá)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Nhận diện khuôn mặt nếu bật
            if self.recognition_enabled:
                face_crop = gray[y:y+h, x:x+w]
                name = self.recognize_face(face_crop)
                cv2.putText(frame, name, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f'Face {i+1}', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # ROI cho mặt
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes trong vùng mặt
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
            total_eyes += len(eyes)
            
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
                cv2.putText(roi_color, 'Eye', (ex, ey-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # Detect smile trong vùng mặt
            smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
            total_smiles += len(smiles)
            
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
                cv2.putText(roi_color, 'Smile', (sx, sy-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Lưu dataset nếu được bật
        if self.save_dataset:
            self.save_face_dataset(gray, faces)
        
        return frame, len(faces), total_eyes, total_smiles
    
    def setup_gui(self):
        """Tạo GUI controls"""
        self.root = tk.Tk()
        self.root.title("Face Dataset Control")
        self.root.geometry("350x250")
        
        # Dropdown cho tên người
        tk.Label(self.root, text="Chọn người:").pack(pady=5)
        self.person_var = tk.StringVar(value="DuongBao")
        self.person_combo = ttk.Combobox(self.root, textvariable=self.person_var,
                                        values=["DuongBao", "MinhAnh", "Tung","Hung"],
                                        state="readonly")
        self.person_combo.pack(pady=5)
        
        # Nút bắt đầu/dừng lưu
        self.save_button = tk.Button(self.root, text="Bắt đầu lưu dataset",
                                    command=self.toggle_dataset_saving,
                                    bg="green", fg="white")
        self.save_button.pack(pady=5)
        
        # Nút train model
        self.train_button = tk.Button(self.root, text="Train Model",
                                     command=self.train_model,
                                     bg="blue", fg="white")
        self.train_button.pack(pady=5)
        
        # Nút bật/tắt nhận diện
        self.recognition_button = tk.Button(self.root, text="Bật nhận diện",
                                           command=self.toggle_recognition,
                                           bg="orange", fg="white")
        self.recognition_button.pack(pady=5)
        
        # Status label
        self.status_label = tk.Label(self.root, text="Chưa lưu dataset")
        self.status_label.pack(pady=5)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def toggle_dataset_saving(self):
        """Bật/tắt lưu dataset"""
        self.save_dataset = not self.save_dataset
        self.selected_person = self.person_var.get()
        
        if self.save_dataset:
            self.save_button.config(text="Dừng lưu dataset", bg="red")
            self.status_label.config(text=f"Đang lưu cho: {self.selected_person}")
        else:
            self.save_button.config(text="Bắt đầu lưu dataset", bg="green")
            self.status_label.config(text="Chưa lưu dataset")
    
    def load_or_train_model(self):
        """Load model nếu có, không thì train mới"""
        model_path = 'face_model.pkl'
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.face_recognizer = data['model']
                    self.label_encoder = data['encoder']
                print("Model loaded successfully")
            except:
                print("Failed to load model")
    
    def train_model(self):
        """Train face recognition model"""
        print("Training face recognition model...")
        
        faces = []
        labels = []
        
        # Đọc dữ liệu từ dataset
        for person in ["DuongBao", "MinhAnh", "Tung", "Hung"]:
            person_folder = os.path.join('face_dataset', person)
            if os.path.exists(person_folder):
                for filename in os.listdir(person_folder):
                    if filename.endswith('.jpg'):
                        img_path = os.path.join(person_folder, filename)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            faces.append(img.flatten())
                            labels.append(person)
        
        if len(faces) > 0:
            # Train KNN classifier
            self.label_encoder = LabelEncoder()
            encoded_labels = self.label_encoder.fit_transform(labels)
            
            self.face_recognizer = KNeighborsClassifier(n_neighbors=3)
            self.face_recognizer.fit(faces, encoded_labels)
            
            # Lưu model
            with open('face_model.pkl', 'wb') as f:
                pickle.dump({
                    'model': self.face_recognizer,
                    'encoder': self.label_encoder
                }, f)
            
            print(f"Model trained with {len(faces)} images")
            self.status_label.config(text=f"Model trained: {len(faces)} images")
        else:
            print("No training data found")
            self.status_label.config(text="No training data found")
    
    def toggle_recognition(self):
        """Bật/tắt nhận diện"""
        if self.face_recognizer is None:
            print("Please train model first")
            return
            
        self.recognition_enabled = not self.recognition_enabled
        
        if self.recognition_enabled:
            self.recognition_button.config(text="Tắt nhận diện", bg="red")
            print("Face recognition enabled")
        else:
            self.recognition_button.config(text="Bật nhận diện", bg="orange")
            print("Face recognition disabled")
    
    def recognize_face(self, face_crop):
        """Nhận diện khuôn mặt"""
        if self.face_recognizer is None or not self.recognition_enabled:
            return "Other"
        
        try:
            face_resized = cv2.resize(face_crop, (100, 100))
            face_vector = face_resized.flatten().reshape(1, -1)
            
            prediction = self.face_recognizer.predict(face_vector)
            confidence = self.face_recognizer.predict_proba(face_vector).max()
            
            if confidence > 0.6:  # Ngưỡng tin cậy
                name = self.label_encoder.inverse_transform(prediction)[0]
                return f"{name} ({confidence:.2f})"
            else:
                return "Other"
        except:
            return "Other"
    
    def on_closing(self):
        """Xử lý khi đóng GUI"""
        self.save_dataset = False
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()
        
    def create_dataset_folders(self):
        """Tạo thư mục dataset"""
        if not os.path.exists('face_dataset'):
            os.makedirs('face_dataset')
        
        # Tạo folder cho từng người
        for person in ["DuongBao", "MinhAnh", "Tung","Hung"]:
            person_folder = os.path.join('face_dataset', person)
            if not os.path.exists(person_folder):
                os.makedirs(person_folder)
    
    def save_face_dataset(self, gray_frame, faces):
        """Lưu dataset cho người được chọn"""
        if len(faces) > 0:
            # Chỉ lưu khuôn mặt đầu tiên
            x, y, w, h = faces[0]
            
            # Sử dụng tên người được chọn
            person_folder = os.path.join('face_dataset', self.selected_person)
            
            # Crop khuôn mặt
            face_crop = gray_frame[y:y+h, x:x+w]
            
            # Resize về kích thước chuẩn
            face_resized = cv2.resize(face_crop, (100, 100))
            
            # Lưu ảnh với timestamp
            if self.selected_person not in self.dataset_counter:
                self.dataset_counter[self.selected_person] = 0
                
            self.dataset_counter[self.selected_person] += 1
            
            # Chỉ lưu mỗi 5 frame
            if self.dataset_counter[self.selected_person] % 5 == 0:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'{self.selected_person}_{timestamp}_{self.dataset_counter[self.selected_person]}.jpg'
                filepath = os.path.join(person_folder, filename)
                cv2.imwrite(filepath, face_resized)
                print(f'Saved: {filepath}')
    
    def add_info_panel(self, frame, num_faces, num_eyes, num_smiles):
        """Thêm panel thông tin lên frame"""
        # Tạo panel thông tin
        info_panel = np.zeros((120, frame.shape[1], 3), dtype=np.uint8)
        
        # Thông tin detection
        cv2.putText(info_panel, f'HAAR CASCADE FACE DETECTION', (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(info_panel, f'Faces: {num_faces}', (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(info_panel, f'Eyes: {num_eyes}', (150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        cv2.putText(info_panel, f'Smiles: {num_smiles}', (280, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Dataset status
        if self.save_dataset:
            dataset_text = f'Saving: {self.selected_person}'
            dataset_color = (0, 255, 0)
        else:
            dataset_text = 'Dataset: OFF'
            dataset_color = (0, 0, 255)
        cv2.putText(info_panel, dataset_text, (420, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, dataset_color, 2)
        
        # Recognition status
        if self.recognition_enabled:
            recog_text = 'Recognition: ON'
            recog_color = (255, 255, 0)
        else:
            recog_text = 'Recognition: OFF'
            recog_color = (128, 128, 128)
        cv2.putText(info_panel, recog_text, (420, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, recog_color, 2)
        
        # Hướng dẫn
        cv2.putText(info_panel, f'Q:quit | S:screenshot | D:toggle dataset', (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.putText(info_panel, f'XML: frontalface | eye | smile', (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Ghép panel với frame
        combined_frame = np.vstack((info_panel, frame))
        return combined_frame
    
    def run(self):
        """Chạy camera detection với GUI"""
        print("Starting camera face detection with GUI...")
        print("Controls:")
        print("   - Press 'q' to quit")
        print("   - Press 's' to save screenshot")
        print("   - Use GUI to control dataset saving")
        print("Detection features: Face (Green), Eyes (Blue), Smile (Red)")
        
        frame_count = 0
        
        while True:
            # Cập nhật GUI
            self.root.update()
            
            ret, frame = self.cap.read()
            
            if not ret:
                print("Cannot read from camera!")
                break
            
            # Flip frame horizontally (mirror effect)
            frame = cv2.flip(frame, 1)
            
            # Detect features
            processed_frame, num_faces, num_eyes, num_smiles = self.detect_features(frame)
            
            # Add info panel
            final_frame = self.add_info_panel(processed_frame, num_faces, num_eyes, num_smiles)
            
            # Hiển thị frame
            cv2.imshow('Multi-Face Detection - Haar Cascade', final_frame)
            
            # Xử lý phím
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                filename = f'face_detection_frame_{frame_count}.jpg'
                cv2.imwrite(filename, final_frame)
                print(f"Screenshot saved: {filename}")
                frame_count += 1
        
        # Cleanup
        self.on_closing()

def main():
    """Main function"""
    try:
        detector = CameraFaceDetector()
        detector.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your camera is connected and not being used by another application.")

if __name__ == "__main__":
    main()
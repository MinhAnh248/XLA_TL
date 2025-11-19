import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import time
import shutil

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System - Professional Edition")
        self.root.geometry("1200x800")
        
        # Haar cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        self.known_faces = {}
        self.load_known_faces()
        
        self.current_image = None
        self.detected_faces = []
        self.collecting = False
        self.collect_name = ""
        self.collect_count = 0
        self.camera_running = False
        self.cap = None
        self.show_fps = False
        self.fps_counter = 0
        self.fps_start_time = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        self.root.configure(bg='#2c3e50')
        
        main_frame = tk.Frame(self.root, bg='#2c3e50', padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = tk.Label(main_frame, text="Face Recognition System", 
                              font=('Arial', 24, 'bold'), fg='#ecf0f1', bg='#2c3e50')
        title_label.pack(pady=(0, 20))
        
        container = tk.Frame(main_frame, bg='#2c3e50')
        container.pack(fill=tk.BOTH, expand=True)
        
        left_frame = tk.Frame(container, bg='#34495e', padx=20, pady=20)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        control_label = tk.Label(left_frame, text="Control Panel", 
                                font=('Arial', 16, 'bold'), fg='#ecf0f1', bg='#34495e')
        control_label.pack(pady=(0, 20))
        
        button_style = {'font': ('Arial', 12), 'width': 18, 'height': 2, 
                       'bg': '#3498db', 'fg': 'white', 'relief': 'flat',
                       'activebackground': '#2980b9', 'activeforeground': 'white'}
        
        tk.Button(left_frame, text="📁 Chọn ảnh", command=self.load_image, **button_style).pack(pady=5)
        tk.Button(left_frame, text="🔍 Phát hiện khuôn mặt", command=self.detect_faces, **button_style).pack(pady=5)
        tk.Button(left_frame, text="📷 Camera", command=self.start_camera, **button_style).pack(pady=5)
        tk.Button(left_frame, text="⏹️ Stop Camera", command=self.stop_camera, **button_style).pack(pady=5)
        tk.Button(left_frame, text="👤 Register User", command=self.collect_dataset, **button_style).pack(pady=5)
        tk.Button(left_frame, text="💿 Lưu dataset", command=self.save_faces, **button_style).pack(pady=5)
        tk.Button(left_frame, text="📊 FPS", command=self.toggle_fps, **button_style).pack(pady=5)
        tk.Button(left_frame, text="🔄 Refresh List", command=self.load_known_faces, **button_style).pack(pady=5)
        tk.Button(left_frame, text="🗑️ Delete User", command=self.delete_user, **button_style).pack(pady=5)
        tk.Button(left_frame, text="❌ Quit", command=self.quit_app, **button_style).pack(pady=5)
        
        right_frame = tk.Frame(container, bg='#ecf0f1', padx=20, pady=20)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        display_label = tk.Label(right_frame, text="Display Area", 
                                font=('Arial', 16, 'bold'), fg='#2c3e50', bg='#ecf0f1')
        display_label.pack(pady=(0, 10))
        
        image_frame = tk.Frame(right_frame, bg='#bdc3c7', relief='solid', bd=2)
        image_frame.pack(pady=10)
        
        self.image_label = tk.Label(image_frame, bg='#ecf0f1', 
                                   text="No image loaded", font=('Arial', 14),
                                   fg='#7f8c8d')
        self.image_label.pack(padx=10, pady=10)
        
        self.image_label.config(width=640, height=480)
        self.image_label.pack_propagate(False)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.current_image = cv2.imread(file_path)
            self.display_image(self.current_image)
            
    def display_image(self, cv_image):
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_image, (640, 480))
        pil_image = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(pil_image)
        self.image_label.configure(image=photo, text="", width=640, height=480)
        self.image_label.image = photo
        
    def detect_faces(self):
        if self.current_image is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước")
            return
            
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        self.detected_faces = faces
        
        result_image = self.current_image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            face_gray = gray[y:y+h, x:x+w]
            
            upper_face = face_gray[:h//2, :]
            eyes = self.eye_cascade.detectMultiScale(upper_face, 1.2, 5)
            for (ex, ey, ew, eh) in eyes:
                if ey < h//3:
                    cv2.rectangle(result_image, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
                
            mouths = self.mouth_cascade.detectMultiScale(face_gray, 1.7, 11)
            for (mx, my, mw, mh) in mouths:
                cv2.rectangle(result_image, (x+mx, y+my), (x+mx+mw, y+my+mh), (0, 0, 255), 2)
            
        self.display_image(result_image)
            
    def start_camera(self):
        if not self.camera_running:
            self.camera_running = True
            self.cap = cv2.VideoCapture(0)
            self.fps_start_time = time.time()
            self.fps_counter = 0
            self.update_frame()
    
    def stop_camera(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        
    def update_frame(self):
        if self.camera_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.fps_counter += 1
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    face_img = frame[y:y+h, x:x+w]
                    name = self.recognize_face(face_img)
                    cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    face_gray = gray[y:y+h, x:x+w]
                    
                    upper_face = face_gray[:h//2, :]
                    eyes = self.eye_cascade.detectMultiScale(upper_face, 1.2, 5)
                    for (ex, ey, ew, eh) in eyes:
                        if ey < h//3:
                            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
                        
                    mouths = self.mouth_cascade.detectMultiScale(face_gray, 1.7, 11)
                    for (mx, my, mw, mh) in mouths:
                        cv2.rectangle(frame, (x+mx, y+my), (x+mx+mw, y+my+mh), (0, 0, 255), 2)
                
                if self.show_fps:
                    elapsed = time.time() - self.fps_start_time
                    if elapsed > 0:
                        fps = self.fps_counter / elapsed
                        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                self.display_image(frame)
                self.root.after(30, self.update_frame)
            else:
                self.stop_camera()
    
    def toggle_fps(self):
        self.show_fps = not self.show_fps
        self.fps_start_time = time.time()
        self.fps_counter = 0
    
    def delete_user(self):
        if not os.path.exists("face_dataset"):
            messagebox.showwarning("Cảnh báo", "Không có dataset")
            return
            
        users = [d for d in os.listdir("face_dataset") if os.path.isdir(os.path.join("face_dataset", d))]
        if not users:
            messagebox.showwarning("Cảnh báo", "Không có user nào")
            return
            
        user_list = "\n".join([f"{i+1}. {user}" for i, user in enumerate(users)])
        choice = tk.simpledialog.askstring("Xóa User", f"Chọn user cần xóa:\n{user_list}\n\nNhập tên:")
        
        if choice and choice in users:
            result = messagebox.askyesno("Xác nhận", f"Xóa user '{choice}'?")
            if result:
                shutil.rmtree(os.path.join("face_dataset", choice))
                messagebox.showinfo("Thành công", f"Xóa user '{choice}' thành công")
                self.load_known_faces()
    
    def quit_app(self):
        self.stop_camera()
        self.root.quit()
        
    def save_faces(self):
        if self.current_image is None or len(self.detected_faces) == 0:
            messagebox.showwarning("Cảnh báo", "Vui lòng phát hiện khuôn mặt trước")
            return
            
        dataset_dir = "face_dataset"
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            
        name = tk.simpledialog.askstring("Tên", "Nhập tên người:")
        if not name:
            return
            
        person_dir = os.path.join(dataset_dir, name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
            
        for i, (x, y, w, h) in enumerate(self.detected_faces):
            face = self.current_image[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (160, 160))
            
            existing_files = len([f for f in os.listdir(person_dir) if f.endswith('.jpg')])
            filename = f"{name}_{existing_files + i + 1}.jpg"
            filepath = os.path.join(person_dir, filename)
            
            cv2.imwrite(filepath, face_resized)
            
        messagebox.showinfo("Thành công", f"Lưu {len(self.detected_faces)} khuôn mặt vào face_dataset/{name}/")
        self.load_known_faces()
        
    def collect_dataset(self):
        name = tk.simpledialog.askstring("Tên", "Nhập tên người:")
        if not name:
            return
            
        self.collect_name = name
        self.collect_count = 0
        self.collecting = True
        
        dataset_dir = os.path.join("face_dataset", name)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            
        cap = cv2.VideoCapture(0)
        
        def collect_frame():
            ret, frame = cap.read()
            if ret and self.collecting:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    if self.collect_count % 10 == 0 and self.collect_count < 500:
                        face = frame[y:y+h, x:x+w]
                        face_resized = cv2.resize(face, (160, 160))
                        filename = f"{name}_{self.collect_count//10 + 1}.jpg"
                        filepath = os.path.join(dataset_dir, filename)
                        cv2.imwrite(filepath, face_resized)
                        
                    self.collect_count += 1
                
                cv2.putText(frame, f"Collected: {self.collect_count//10}/50", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                self.display_image(frame)
                
                if self.collect_count < 500:
                    self.root.after(30, collect_frame)
                else:
                    cap.release()
                    self.collecting = False
                    messagebox.showinfo("Hoàn thành", f"Thu thập xong 50 ảnh cho {name}")
                    self.load_known_faces()
            else:
                cap.release()
                
        collect_frame()
        
    def load_known_faces(self):
        self.known_faces = {}
        face_dataset_dir = "face_dataset"
        
        if not os.path.exists(face_dataset_dir):
            return
            
        for person_name in os.listdir(face_dataset_dir):
            person_dir = os.path.join(face_dataset_dir, person_name)
            if os.path.isdir(person_dir):
                face_encodings = []
                for img_file in os.listdir(person_dir):
                    if img_file.endswith(('.jpg', '.png')):
                        img_path = os.path.join(person_dir, img_file)
                        img = cv2.imread(img_path)
                        if img is not None:
                            face_encodings.append(img)
                            
                if face_encodings:
                    self.known_faces[person_name] = face_encodings
                    
    def recognize_face(self, face_img):
        face_resized = cv2.resize(face_img, (160, 160))
        
        for name, known_faces in self.known_faces.items():
            for known_face in known_faces:
                hist1 = cv2.calcHist([face_resized], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
                hist2 = cv2.calcHist([known_face], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
                
                similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                if similarity > 0.7:
                    return name
                    
        return "Unknown"

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
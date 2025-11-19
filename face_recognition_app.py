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
import stat

# Optional: facenet-pytorch for embeddings
try:
    import torch
    from torchvision import transforms
    from facenet_pytorch import InceptionResnetV1
except Exception:
    torch = None
    transforms = None
    InceptionResnetV1 = None
try:
    from pymongo import MongoClient
    from bson.binary import Binary
except Exception:
    MongoClient = None
    Binary = None

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

        # FaceNet / Embedding model (optional)
        self.use_facenet = False
        self.device = None
        self.resnet = None
        self.embeddings = {}
        self.embedding_file = "face_embeddings.pkl"
        # MongoDB settings (use env var MONGO_URI if provided)
        self.mongo_client = None
        self.mongo_db = None
        self.mongo_col = None
        # Default Mongo URI: try local MongoDB if no env var provided
        self.mongo_uri = os.environ.get('MONGO_URI') or "mongodb://localhost:27017/face_recognition"
        if MongoClient is not None and self.mongo_uri:
            try:
                self.mongo_client = MongoClient(self.mongo_uri)
                # get_database() uses the database from URI if provided; otherwise use 'face_recognition'
                try:
                    self.mongo_db = self.mongo_client.get_database()
                except Exception:
                    self.mongo_db = self.mongo_client['face_recognition']
                self.mongo_col = self.mongo_db.get_collection('face_embeddings')
            except Exception:
                self.mongo_client = None
                self.mongo_db = None
                self.mongo_col = None

        if InceptionResnetV1 is not None:
            try:
                self.device = torch.device('cuda' if torch and torch.cuda.is_available() else 'cpu')
                self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
                self.use_facenet = True
                # try to load existing embeddings
                self.load_embeddings()
            except Exception:
                self.use_facenet = False
        
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
        tk.Button(left_frame, text="💾 Lưu lên Mongo", command=self.save_faces, **button_style).pack(pady=5)
        tk.Button(left_frame, text="📊 FPS", command=self.toggle_fps, **button_style).pack(pady=5)
        tk.Button(left_frame, text="🔄 Refresh List", command=self.load_known_faces, **button_style).pack(pady=5)
        tk.Button(left_frame, text="🧠 Build Embeddings", command=self.on_build_embeddings, **button_style).pack(pady=5)
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
        # Show selection dialog with names from Mongo
        if self.mongo_col is None:
            messagebox.showwarning("Cảnh báo", "MongoDB chưa được kết nối. Không thể xóa user.")
            return

        try:
            docs = list(self.mongo_col.find({}, {'name': 1}))
            users = [d['name'] for d in docs if 'name' in d]
        except Exception:
            messagebox.showerror("Lỗi", "Không thể truy vấn MongoDB để lấy danh sách user")
            return

        if not users:
            messagebox.showwarning("Cảnh báo", "Không có user nào trong MongoDB")
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("Xóa User")
        dlg.geometry("300x300")
        tk.Label(dlg, text="Chọn user để xóa:").pack(pady=8)
        lb = tk.Listbox(dlg)
        lb.pack(fill=tk.BOTH, expand=True, padx=10)
        for u in users:
            lb.insert(tk.END, u)

        def on_confirm():
            sel = lb.curselection()
            if not sel:
                messagebox.showwarning("Cảnh báo", "Vui lòng chọn một user")
                return
            name = lb.get(sel[0])
            if not messagebox.askyesno("Xác nhận", f"Xóa user '{name}'? Đây là hành động không thể hoàn tác."):
                return

            try:
                self.mongo_col.delete_one({'name': name})
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể xóa trong MongoDB: {e}")
                dlg.destroy()
                return

            messagebox.showinfo("Thành công", f"Xóa user '{name}' thành công (MongoDB)")
            dlg.destroy()
            # reload embeddings from Mongo
            self.load_embeddings()

        btn_frame = tk.Frame(dlg)
        btn_frame.pack(pady=8)
        tk.Button(btn_frame, text="Xóa", command=on_confirm, bg='#e74c3c', fg='white', width=10).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_frame, text="Hủy", command=dlg.destroy, width=10).pack(side=tk.RIGHT, padx=6)
    
    def quit_app(self):
        self.stop_camera()
        self.root.quit()
        
    def save_faces(self):
        # Save computed embedding(s) directly to Mongo for the current detected faces
        if self.current_image is None or len(self.detected_faces) == 0:
            messagebox.showwarning("Cảnh báo", "Vui lòng phát hiện khuôn mặt trước")
            return

        if not self.use_facenet or self.resnet is None:
            messagebox.showwarning("Cảnh báo", "FaceNet chưa sẵn sàng. Hãy cài đặt 'facenet-pytorch' và 'torch'.")
            return

        if self.mongo_col is None:
            messagebox.showwarning("Cảnh báo", "MongoDB chưa được kết nối. Không thể lưu embeddings.")
            return

        name = tk.simpledialog.askstring("Tên", "Nhập tên người:")
        if not name:
            return

        vecs = []
        for (x, y, w, h) in self.detected_faces:
            face = self.current_image[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (160, 160))
            emb = self._compute_embedding(face_resized)
            if emb is not None:
                vecs.append(emb)

        if not vecs:
            messagebox.showwarning("Lỗi", "Không thể tính embedding từ khuôn mặt đã phát hiện")
            return

        arr = np.stack(vecs)
        try:
            buf = __import__('io').BytesIO()
            np.save(buf, arr, allow_pickle=False)
            buf.seek(0)
            bin_data = Binary(buf.read())
            doc = {'name': name, 'embeddings': bin_data, 'dim': int(arr.shape[1]), 'count': int(arr.shape[0])}
            self.mongo_col.replace_one({'name': name}, doc, upsert=True)
            messagebox.showinfo("Thành công", f"Đã lưu embeddings của '{name}' lên MongoDB")
            # reload embeddings into memory
            self.load_embeddings()
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu embeddings lên MongoDB: {e}")
        
    def collect_dataset(self):
        # Collect a few frames and compute embeddings in-memory, then save to Mongo
        name = tk.simpledialog.askstring("Tên", "Nhập tên người:")
        if not name:
            return

        if not self.use_facenet or self.resnet is None:
            messagebox.showwarning("Cảnh báo", "FaceNet chưa sẵn sàng. Hãy cài đặt 'facenet-pytorch' và 'torch'.")
            return

        if self.mongo_col is None:
            messagebox.showwarning("Cảnh báo", "MongoDB chưa được kết nối. Không thể lưu embeddings.")
            return

        self.collect_name = name
        self.collect_count = 0
        self.collecting = True

        cap = cv2.VideoCapture(0)
        collected_vecs = []

        def collect_frame():
            ret, frame = cap.read()
            if ret and self.collecting:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    if self.collect_count % 10 == 0 and len(collected_vecs) < 50:
                        face = frame[y:y+h, x:x+w]
                        face_resized = cv2.resize(face, (160, 160))
                        emb = self._compute_embedding(face_resized)
                        if emb is not None:
                            collected_vecs.append(emb)

                    self.collect_count += 1

                cv2.putText(frame, f"Collected: {len(collected_vecs)}/50", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                self.display_image(frame)

                if len(collected_vecs) < 50 and self.collect_count < 500:
                    self.root.after(30, collect_frame)
                else:
                    cap.release()
                    self.collecting = False
                    if not collected_vecs:
                        messagebox.showwarning("Lỗi", "Không thu thập được embedding nào")
                        return
                    arr = np.stack(collected_vecs)
                    try:
                        buf = __import__('io').BytesIO()
                        np.save(buf, arr, allow_pickle=False)
                        buf.seek(0)
                        bin_data = Binary(buf.read())
                        doc = {'name': name, 'embeddings': bin_data, 'dim': int(arr.shape[1]), 'count': int(arr.shape[0])}
                        self.mongo_col.replace_one({'name': name}, doc, upsert=True)
                        messagebox.showinfo("Hoàn thành", f"Đã lưu embeddings cho {name} lên MongoDB")
                        self.load_embeddings()
                    except Exception as e:
                        messagebox.showerror("Lỗi", f"Không thể lưu embeddings lên MongoDB: {e}")
            else:
                cap.release()

        collect_frame()
        
    def load_known_faces(self):
        # No-op: dataset-based known faces deprecated.
        # Use `load_embeddings()` to refresh embeddings from MongoDB instead.
        self.known_faces = {}
        self.load_embeddings()
                    
    def recognize_face(self, face_img):
        # Only use FaceNet embeddings loaded from MongoDB
        if not self.use_facenet or not self.embeddings:
            return "Unknown"

        try:
            emb = self._compute_embedding(face_img)
            if emb is None:
                return "Unknown"

            best_name = "Unknown"
            best_dist = float('inf')
            for name, vecs in self.embeddings.items():
                dists = np.linalg.norm(vecs - emb, axis=1)
                min_dist = float(np.min(dists))
                if min_dist < best_dist:
                    best_dist = min_dist
                    best_name = name

            if best_dist < 0.9:
                return best_name
            return "Unknown"
        except Exception:
            return "Unknown"

    def _compute_embedding(self, face_img):
        """Compute FaceNet embedding for a single face image (numpy BGR)"""
        if not self.use_facenet or self.resnet is None:
            return None

        try:
            # convert BGR (OpenCV) to RGB
            rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb).resize((160, 160))

            # to tensor and normalize to [-1, 1]
            if transforms is not None:
                tf = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
                img_t = tf(pil).unsqueeze(0).to(self.device)
            else:
                # basic numpy fallback
                arr = np.asarray(pil).astype(np.float32) / 255.0
                arr = (arr - 0.5) / 0.5
                arr = np.transpose(arr, (2, 0, 1))
                img_t = torch.from_numpy(arr).unsqueeze(0).to(self.device)

            with torch.no_grad():
                emb = self.resnet(img_t)
            emb = emb.cpu().numpy().reshape(-1)
            return emb
        except Exception:
            return None

    def build_embeddings(self):
        # Deprecated: building from local dataset is removed. Embeddings must be stored in Mongo.
        return False

    def load_embeddings(self):
        # Load embeddings exclusively from MongoDB
        self.embeddings = {}
        if self.mongo_col is None:
            return False
        try:
            docs = list(self.mongo_col.find({}))
            data = {}
            for doc in docs:
                name = doc.get('name')
                bin_data = doc.get('embeddings')
                if name and bin_data:
                    buf = __import__('io').BytesIO(bin_data)
                    buf.seek(0)
                    arr = np.load(buf, allow_pickle=False)
                    data[name] = arr
            if data:
                self.embeddings = data
                return True
        except Exception:
            self.embeddings = {}
        return False

    def _safe_rmtree(self, path):
        """Recursively remove a directory on Windows, clearing read-only flags if needed."""
        if not os.path.exists(path):
            return

        # Walk files and ensure writable, then remove
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                filename = os.path.join(root, name)
                try:
                    os.chmod(filename, stat.S_IWRITE)
                except Exception:
                    pass
            for name in dirs:
                dirname = os.path.join(root, name)
                try:
                    os.chmod(dirname, stat.S_IWRITE)
                except Exception:
                    pass

        # Try shutil.rmtree now
        try:
            shutil.rmtree(path)
        except PermissionError:
            # Second attempt: remove files individually
            for root, dirs, files in os.walk(path, topdown=False):
                for name in files:
                    filename = os.path.join(root, name)
                    try:
                        os.chmod(filename, stat.S_IWRITE)
                        os.remove(filename)
                    except Exception:
                        pass
                for name in dirs:
                    dirname = os.path.join(root, name)
                    try:
                        os.chmod(dirname, stat.S_IWRITE)
                        os.rmdir(dirname)
                    except Exception:
                        pass
            # finally try to remove the root
            if os.path.exists(path):
                try:
                    os.rmdir(path)
                except Exception as e:
                    raise e

    def on_build_embeddings(self):
        """GUI handler to build embeddings and notify user."""
        messagebox.showinfo("Không khả dụng", "Tính năng này đã bị vô hiệu: embeddings giờ chỉ lưu trực tiếp lên MongoDB khi Register/Save.")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
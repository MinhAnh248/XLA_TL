from flask import Flask, render_template, Response, jsonify, request
import cv2
import time
import threading
from face_detection import FaceDetection
from face_recognizer import FaceRecognizer

class CameraThread:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 24)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.running = True
        t = threading.Thread(target=self.update, daemon=True)
        t.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def get_frame(self):
        frame = self.frame
        self.frame = None
        return frame

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()

app = Flask(__name__)

# Khởi tạo face detection và recognition
face_detector = FaceDetection()
face_recognizer = FaceRecognizer()
face_recognizer.load_model()

# Biến global
face_count = 0
detection_active = True
auto_save = False
save_interval = 0.5  # Mặc định 0.5 giây
last_save_time = 0
camera = None

def get_camera():
    """Lấy camera instance tối ưu"""
    global camera
    if camera is None:
        camera = CameraThread()
    return camera

def release_camera():
    """Giải phóng camera khi Flask tắt"""
    global camera
    if camera is not None:
        camera.stop()
        camera = None
        print("✅ Camera released")

def generate_frames():
    """Generator để stream video"""
    global face_count, detection_active, auto_save, last_save_time
    
    cam = get_camera()
    FRAME_DELAY = 1 / 24
    frame_count = 0
    
    while True:
        start = time.time()
        frame = cam.get_frame()
        if frame is None:
            time.sleep(0.005)
            continue
        
        frame_count += 1
        
        if detection_active and frame_count % 3 == 0:
            faces = face_detector.detect_faces(frame)
            face_count = len(faces)
            
            current_time = time.time()
            if auto_save and len(faces) > 0 and (current_time - last_save_time) > save_interval:
                face_detector.save_face_samples(frame, faces)
                last_save_time = current_time
            
            frame = draw_faces_with_recognition(frame, faces)
            cv2.putText(frame, f'Faces: {face_count}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        if not ret:
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        elapsed = time.time() - start
        if elapsed < FRAME_DELAY:
            time.sleep(FRAME_DELAY - elapsed)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_auto_save', methods=['POST'])
def toggle_auto_save():
    global auto_save
    auto_save = not auto_save
    return jsonify({'auto_save': auto_save})

@app.route('/set_save_interval', methods=['POST'])
def set_save_interval():
    global save_interval
    data = request.get_json()
    save_interval = float(data.get('interval', 0.5))  # Hỗ trợ số thập phân
    return jsonify({'save_interval': save_interval})

@app.route('/capture', methods=['POST'])
def capture_image():
    """Chụp ảnh thủ công"""
    cam = get_camera()
    frame = cam.get_frame()
    
    if frame is not None:
        faces = face_detector.detect_faces(frame)
        if len(faces) > 0:
            saved_files = face_detector.save_face_samples(frame, faces)
            return jsonify({
                'success': True, 
                'faces_saved': len(faces),
                'files': saved_files
            })
        else:
            return jsonify({'success': False, 'message': 'No faces detected'})
    
    return jsonify({'success': False, 'message': 'Camera error'})

@app.route('/get_samples')
def get_samples():
    samples = face_detector.get_recent_samples(12)
    return jsonify({'samples': samples})

@app.route('/dataset/<person>/<filename>')
def serve_dataset_image(person, filename):
    """Phục vụ ảnh từ dataset"""
    from flask import send_from_directory
    return send_from_directory(f'dataset/{person}', filename)

@app.route('/recognition')
def recognition():
    return render_template('recognition.html')

@app.route('/recognition_feed')
def recognition_feed():
    return Response(generate_recognition_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognition_stats')
def recognition_stats():
    return jsonify({
        'face_count': face_count,
        'recognized_count': getattr(generate_recognition_frames, 'recognized_count', 0)
    })

def generate_recognition_frames():
    """Generator chỉ để nhận diện"""
    global face_count
    cam = get_camera()
    FRAME_DELAY = 1 / 24
    frame_count = 0
    
    while True:
        start = time.time()
        frame = cam.get_frame()
        if frame is None:
            time.sleep(0.005)
            continue
        
        frame_count += 1
        
        if frame_count % 3 == 0:
            faces = face_detector.detect_faces(frame)
            face_count = len(faces)
            recognized_count = 0
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (160, 160))
                name, confidence = face_recognizer.recognize_face(face_resized)
                
                if name != "Unknown":
                    recognized_count += 1
                    if name == 'MinhAnh':
                        color = (0, 255, 0)
                    elif name == 'ThuyTien':
                        color = (255, 0, 255)
                    elif name == 'ThinhNho':
                        color = (255, 165, 0)
                    elif name == 'DuongBao':
                        color = (0, 165, 255)
                    else:
                        color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, name, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(frame, confidence, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            generate_recognition_frames.recognized_count = recognized_count
            cv2.putText(frame, f'Recognized: {recognized_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        if not ret:
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        elapsed = time.time() - start
        if elapsed < FRAME_DELAY:
            time.sleep(FRAME_DELAY - elapsed)

@app.route('/stats')
def get_stats():
    sample_count = face_detector.get_sample_count()
    return jsonify({
        'face_count': face_count,
        'status': 'active' if detection_active else 'paused',
        'auto_save': auto_save,
        'sample_count': sample_count,
        'save_interval': save_interval,
        'current_person': face_detector.current_person,
        'algorithm': 'Haar Cascade'
    })

@app.route('/set_person', methods=['POST'])
def set_person():
    """Thiết lập người hiện tại"""
    data = request.get_json()
    person_name = data.get('person', 'ThuyTien')
    face_detector.set_current_person(person_name)
    return jsonify({'success': True, 'current_person': person_name})

@app.route('/get_persons')
def get_persons():
    """Lấy danh sách người"""
    persons = face_detector.get_all_persons()
    return jsonify({'persons': persons, 'current_person': face_detector.current_person})

@app.route('/clear_samples', methods=['POST'])
def clear_samples():
    """Xóa tất cả mẫu của người hiện tại"""
    face_detector.clear_samples()
    return jsonify({'success': True, 'message': f'Cleared samples for {face_detector.current_person}'})

@app.route('/train_model', methods=['POST'])
def train_model():
    """Huấn luyện mô hình nhận diện"""
    success = face_recognizer.train_model()
    if success:
        face_recognizer.load_model()
        return jsonify({'success': True, 'message': 'Model trained successfully'})
    return jsonify({'success': False, 'message': 'No training data found'})

def draw_faces_with_recognition(frame, faces):
    """Vẽ khung và nhận diện khuôn mặt"""
    for i, (x, y, w, h) in enumerate(faces):
        # Cắt vùng khuôn mặt để nhận diện
        face_roi = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (160, 160))
        
        # Nhận diện
        name, confidence = face_recognizer.recognize_face(face_resized)
        
        # Chỉ hiển thị khi nhận diện được
        if name != "Unknown":
            # Chọn màu khung theo tên
            if name == 'MinhAnh':
                color = (0, 255, 0)  # Xanh lá
            elif name == 'ThuyTien':
                color = (255, 0, 255)  # Tím
            elif name == 'DuongBao':
                color = (255, 165, 0)  # Cam
            else:
                color = (0, 255, 0)
            
            # Vẽ khung
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Vẽ tên và độ tin cậy
            cv2.putText(frame, f'{name}', (x, y-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f'{confidence}', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Phát hiện các bộ phận khuôn mặt
            face_detector.detect_facial_features(frame, face_resized, x, y)
    
    return frame

def draw_faces_simple(frame, faces):
    """Vẽ khung đơn giản không nhận diện"""
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    global detection_active
    detection_active = not detection_active
    return jsonify({'detection_active': detection_active})

@app.route('/reset_camera', methods=['POST'])
def reset_camera():
    """Reset camera thủ công"""
    release_camera()
    get_camera()
    return jsonify({'success': True, 'message': 'Camera reset successfully'})

if __name__ == '__main__':
    import atexit
    atexit.register(release_camera)
    
    print("Starting Face Detection & Recognition System...")
    print("Dataset structure: dataset/[PersonName]/")
    print("Auto-loading recognition model...")
    face_recognizer.load_model()
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
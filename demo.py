import cv2
import time
from face_detection import FaceDetection
from face_recognizer import FaceRecognizer

class FaceRecognitionDemo:
    def __init__(self):
        self.face_detector = FaceDetection()
        self.face_recognizer = FaceRecognizer()
        self.face_recognizer.load_model()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    def run_demo(self):
        print("Face Recognition Demo")
        print("Press 'q' to quit")
        print("Press 's' to save current faces")
        print("Press 't' to train model")
        print("Press '1' for ThuyTien, '2' for MinhAnh, '3' for ThinhNho")
        
        current_person = "ThuyTien"
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Detect faces
            faces = self.face_detector.detect_faces(frame)
            
            # Draw recognition results
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (160, 160))
                
                name, confidence = self.face_recognizer.recognize_face(face_resized)
                
                if name != "Unknown":
                    if name == 'MinhAnh':
                        color = (0, 255, 0)
                    elif name == 'ThuyTien':
                        color = (255, 0, 255)
                    elif name == 'ThinhNho':
                        color = (255, 165, 0)
                    else:
                        color = (0, 255, 0)
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f'{name}', (x, y-30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(frame, f'{confidence}', (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Phát hiện các bộ phận khuôn mặt
                    self.face_detector.detect_facial_features(frame, face_roi, x, y)
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 128, 128), 1)
            
            # Show info
            cv2.putText(frame, f'Current: {current_person}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f'Faces: {len(faces)}', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show registered people
            persons = self.face_detector.get_all_persons()
            cv2.putText(frame, f'Registered: {", ".join(persons)}', (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Face Recognition Demo', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if len(faces) > 0:
                    self.face_detector.set_current_person(current_person)
                    saved = self.face_detector.save_face_samples(frame, faces)
                    print(f"Saved {len(saved)} samples for {current_person}")
                else:
                    print("No faces detected to save")
            elif key == ord('t'):
                print("Training model...")
                success = self.face_recognizer.train_model()
                if success:
                    print("Model trained successfully!")
                else:
                    print("No training data found")
            elif key == ord('1'):
                current_person = "ThuyTien"
                print(f"Switched to {current_person}")
            elif key == ord('2'):
                current_person = "MinhAnh"
                print(f"Switched to {current_person}")
            elif key == ord('3'):
                current_person = "ThinhNho"
                print(f"Switched to {current_person}")
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    demo = FaceRecognitionDemo()
    demo.run_demo()
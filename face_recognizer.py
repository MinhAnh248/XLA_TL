import cv2
import numpy as np
import os

class FaceRecognizer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.dataset_path = 'dataset'
        self.person_templates = {}
        self.trained = False
        
    def load_templates(self):
        """Load face templates cho tất cả người"""
        self.person_templates = {}
        if not os.path.exists(self.dataset_path):
            return False
            
        for person_name in os.listdir(self.dataset_path):
            person_path = os.path.join(self.dataset_path, person_name)
            if os.path.isdir(person_path):
                templates = []
                for filename in os.listdir(person_path):
                    if filename.endswith('.jpg'):
                        img_path = os.path.join(person_path, filename)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            templates.append(img)
                if templates:
                    self.person_templates[person_name] = templates
        
        self.trained = len(self.person_templates) > 0
        return self.trained
    
    def train_model(self):
        return self.load_templates()
    
    def load_model(self):
        return self.load_templates()
    
    def recognize_face(self, face_img):
        """Nhận diện khuôn mặt với nhiều phương pháp"""
        if not self.trained or len(self.person_templates) == 0:
            return "Unknown", "0%"
        
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
        gray = cv2.resize(gray, (160, 160))
        
        # Cải thiện chất lượng ảnh
        gray = cv2.equalizeHist(gray)
        
        person_scores = {}
        
        for person_name, templates in self.person_templates.items():
            scores = []
            for template in templates:
                # Sử dụng nhiều phương pháp matching
                result1 = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                result2 = cv2.matchTemplate(gray, template, cv2.TM_CCORR_NORMED)
                
                _, max_val1, _, _ = cv2.minMaxLoc(result1)
                _, max_val2, _, _ = cv2.minMaxLoc(result2)
                
                # Kết hợp 2 phương pháp
                combined_score = (max_val1 * 0.7 + max_val2 * 0.3)
                scores.append(combined_score)
            
            # Lấy trung bình của top 3 scores
            scores.sort(reverse=True)
            avg_score = sum(scores[:3]) / min(3, len(scores))
            person_scores[person_name] = avg_score
        
        # Tìm người có điểm cao nhất
        best_match = max(person_scores, key=person_scores.get)
        best_score = person_scores[best_match]
        
        # Kiểm tra khoảng cách với người thứ 2
        sorted_scores = sorted(person_scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            score_diff = sorted_scores[0] - sorted_scores[1]
            if score_diff < 0.1:  # Nếu quá gần nhau thì Unknown
                return "Unknown", f"{best_score*100:.1f}%"
        
        if best_score > 0.65:
            confidence = best_score * 100
            return best_match, f"{confidence:.1f}%"
        else:
            return "Unknown", f"{best_score*100:.1f}%"
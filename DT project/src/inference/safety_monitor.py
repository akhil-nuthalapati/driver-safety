import cv2
import torch
import numpy as np
from torchvision import transforms
import time
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.driver_models import (
    DrowsinessDetector, 
    DistractionDetector,
    EmotionDetector,
    SeatbeltDetector
)
from src.models.risk_engine import RiskScoringEngine

class DriverSafetyMonitor:
    def __init__(self, model_dir='outputs/models'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Initializing Driver Safety Monitor...")
        print(f"   Using device: {self.device}")
        
        # Initialize all detectors
        self.models = {}
        
        # Load distraction detector (your trained model)
        print("\n📦 Loading models:")
        self.models['distraction'] = self._load_model(
            DistractionDetector,
            os.path.join(model_dir, 'distraction_model.pth'),
            num_classes=6,
            required=True  # This is your trained model
        )
        
        # Load other models (optional)
        self.models['drowsiness'] = self._load_model(
            DrowsinessDetector,
            os.path.join(model_dir, 'drowsiness_model.pth'),
            num_classes=3,
            required=False
        )
        
        self.models['emotion'] = self._load_model(
            EmotionDetector,
            os.path.join(model_dir, 'emotion_model.pth'),
            num_classes=5,
            required=False
        )
        
        self.models['seatbelt'] = self._load_model(
            SeatbeltDetector,
            os.path.join(model_dir, 'seatbelt_model.pth'),
            num_classes=2,
            required=False
        )
        
        # Initialize risk engine
        self.risk_engine = RiskScoringEngine()
        
        # Face detector for ROI
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        if self.face_cascade.empty():
            print("⚠️  Face cascade not loaded - face detection disabled")
            self.face_cascade = None
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # For CIFAR-32 models
        self.transform_small = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Classes for display
        self.class_names = {
            'distraction': ['safe_driving', 'texting', 'talking_phone', 
                           'operating_radio', 'drinking', 'reaching_behind'],
            'drowsiness': ['awake', 'drowsy', 'sleeping'],
            'emotion': ['neutral', 'happy', 'angry', 'surprised', 'frustrated'],
            'seatbelt': ['off', 'on']
        }
        
        print("\n✅ System ready! Press 'q' to quit")
    
    def _load_model(self, model_class, path, num_classes=6, required=False):
        """Load trained model"""
        try:
            # Initialize model
            model = model_class(num_classes=num_classes).to(self.device)
            model.eval()
            
            # Try to load weights
            if os.path.exists(path):
                try:
                    # Check if file is a valid checkpoint
                    if os.path.getsize(path) < 1000:  # Too small to be real weights
                        print(f"   ⚠️  {os.path.basename(path)} is a placeholder - using random weights")
                        return model
                    
                    # Load weights
                    state_dict = torch.load(path, map_location=self.device)
                    model.load_state_dict(state_dict)
                    print(f"   ✅ Loaded {os.path.basename(path)} ({os.path.getsize(path)/1024:.1f} KB)")
                    return model
                except Exception as e:
                    print(f"   ⚠️  Error loading {os.path.basename(path)}: {e}")
                    if required:
                        print(f"      Required model failed - continuing with random weights")
                    return model
            else:
                if required:
                    print(f"   ⚠️  Required model {os.path.basename(path)} not found - using random weights")
                else:
                    print(f"   ⚠️  {os.path.basename(path)} not found - using random weights")
                return model
                
        except Exception as e:
            print(f"   ❌ Error creating model: {e}")
            return None
    
    def detect_face(self, frame):
        """Detect face and return ROI"""
        if self.face_cascade is None:
            return None, None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            return frame[y:y+h, x:x+w], (x, y, w, h)
        return None, None
    
    def predict_frame(self, frame):
        """Run all predictions on a single frame"""
        predictions = {}
        
        # Detect face for focused analysis
        face_roi, face_coords = self.detect_face(frame)
        
        # Distraction detection (on full frame)
        if self.models.get('distraction'):
            try:
                # Use smaller transform for CIFAR-trained model
                frame_tensor = self.transform_small(frame).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    dist_output = self.models['distraction'](frame_tensor)
                    dist_prob = torch.softmax(dist_output, dim=1)
                    dist_class_idx = torch.argmax(dist_prob).item()
                    dist_class = self.class_names['distraction'][dist_class_idx]
                    dist_conf = torch.max(dist_prob).item()
                    predictions['distraction'] = (dist_class, dist_conf)
            except Exception as e:
                print(f"Distraction prediction error: {e}")
        
        # Face-based predictions if face detected
        if face_roi is not None:
            try:
                face_tensor = self.transform(face_roi).unsqueeze(0).to(self.device)
                
                # Drowsiness detection
                if self.models.get('drowsiness'):
                    with torch.no_grad():
                        drowsy_output = self.models['drowsiness'](face_tensor)
                        drowsy_prob = torch.softmax(drowsy_output, dim=1)
                        drowsy_class_idx = torch.argmax(drowsy_prob).item()
                        drowsy_class = self.class_names['drowsiness'][drowsy_class_idx]
                        drowsy_conf = torch.max(drowsy_prob).item()
                        predictions['drowsiness'] = (drowsy_class, drowsy_conf)
                
                # Emotion detection
                if self.models.get('emotion'):
                    with torch.no_grad():
                        emotion_output = self.models['emotion'](face_tensor)
                        emotion_prob = torch.softmax(emotion_output, dim=1)
                        emotion_class_idx = torch.argmax(emotion_prob).item()
                        emotion_class = self.class_names['emotion'][emotion_class_idx]
                        emotion_conf = torch.max(emotion_prob).item()
                        predictions['emotion'] = (emotion_class, emotion_conf)
            except Exception as e:
                print(f"Face-based prediction error: {e}")
        
        # Seatbelt detection (on full frame lower half)
        if self.models.get('seatbelt'):
            try:
                # Focus on lower half of frame for seatbelt
                h, w = frame.shape[:2]
                lower_half = frame[h//2:, :]
                belt_tensor = self.transform(lower_half).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    belt_output = self.models['seatbelt'](belt_tensor)
                    belt_prob = torch.softmax(belt_output, dim=1)
                    belt_class_idx = torch.argmax(belt_prob).item()
                    belt_class = self.class_names['seatbelt'][belt_class_idx]
                    belt_conf = torch.max(belt_prob).item()
                    predictions['seatbelt'] = (belt_class, belt_conf)
            except Exception as e:
                print(f"Seatbelt prediction error: {e}")
        
        return predictions
    
    def run_realtime(self, source=0):
        """Run real-time monitoring"""
        print(f"\n🎥 Opening camera source {source}...")
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print("❌ Could not open camera!")
            return
        
        # Set lower resolution for faster processing
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("✅ Camera opened successfully!")
        print("🎥 Starting real-time monitoring... Press 'q' to quit\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Run predictions
            predictions = self.predict_frame(frame)
            
            # Calculate risk (only if we have distraction detection)
            if 'distraction' in predictions:
                risk_info = self.risk_engine.calculate_risk(predictions)
                
                # Check if alert needed
                if self.risk_engine.should_alert(risk_info):
                    alert = self.risk_engine.get_alert_level(risk_info)
                    self.trigger_alert(alert, frame)
            else:
                risk_info = {'total_risk': 0, 'risk_level': 'UNKNOWN'}
            
            # Calculate FPS
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                elapsed = time.time() - self.start_time
                self.fps = self.frame_count / elapsed
            
            # Draw UI
            frame = self.draw_ui(frame, predictions, risk_info)
            
            # Display
            cv2.imshow('Driver Safety Monitor', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Monitoring stopped")
    
    def draw_ui(self, frame, predictions, risk_info):
        """Draw monitoring UI"""
        h, w = frame.shape[:2]
        
        # Risk level color
        risk_colors = {
            'LOW': (0, 255, 0),      # Green
            'MEDIUM': (0, 255, 255),  # Yellow
            'HIGH': (0, 165, 255),    # Orange
            'CRITICAL': (0, 0, 255),   # Red
            'UNKNOWN': (128, 128, 128) # Gray
        }
        
        risk_color = risk_colors.get(risk_info['risk_level'], (255, 255, 255))
        
        # Draw header background
        cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
        
        # Risk meter
        risk_text = f"RISK: {risk_info['risk_level']} ({risk_info['total_risk']:.1f}%)"
        cv2.putText(frame, risk_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, risk_color, 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w-150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw predictions
        y_offset = 55
        for model_name, (cls, conf) in predictions.items():
            # Color based on risk
            if model_name == 'distraction':
                if cls in ['texting', 'talking_phone', 'reaching_behind']:
                    color = (0, 0, 255)  # Red for dangerous
                else:
                    color = (0, 255, 0)  # Green for safe
            else:
                color = (255, 255, 255)  # White for others
            
            text = f"{model_name}: {cls} ({conf:.2f})"
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20
        
        # Draw progress bar for risk
        bar_x, bar_y = w - 220, 65
        bar_w, bar_h = 200, 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), 
                     (50, 50, 50), -1)
        
        fill_w = int(bar_w * (risk_info['total_risk'] / 100))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), 
                     risk_color, -1)
        
        return frame
    
    def trigger_alert(self, alert, frame):
        """Trigger appropriate alert"""
        print(f"🚨 ALERT: {alert['message']}")
        
        if alert['action'] == 'sound_alert':
            # Play sound (implement based on OS)
            if os.name == 'nt':  # Windows
                import winsound
                winsound.Beep(1000, 500)
        
        elif alert['action'] == 'sound_alert + vibration':
            # Sound + visual warning
            h, w = frame.shape[:2]
            cv2.putText(frame, "⚠️ DANGER ⚠️", (w//2-150, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            
            if os.name == 'nt':
                import winsound
                winsound.Beep(2000, 300)
                winsound.Beep(2000, 300)
        
        elif alert['action'] == 'all_alerts + emergency_contact':
            # Critical alert
            h, w = frame.shape[:2]
            cv2.putText(frame, "🚨 EMERGENCY 🚨", (w//2-180, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            
            # Save frame as evidence
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"outputs/emergency_{timestamp}.jpg", frame)
            print(f"📸 Emergency frame saved")

if __name__ == "__main__":
    try:
        monitor = DriverSafetyMonitor()
        monitor.run_realtime()
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
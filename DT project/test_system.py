import sys
import os
import cv2
import torch

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.inference.safety_monitor import DriverSafetyMonitor

def test_models_exist():
    """Check if all model files exist"""
    print("\n🔍 Checking model files...")
    model_files = [
        'outputs/models/distraction_model_best.pth',
        'outputs/models/drowsiness_model_best.pth',
        'outputs/models/emotion_model_best.pth',
        'outputs/models/seatbelt_model_best.pth'
    ]
    
    all_exist = True
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✅ {model_file} - Found")
        else:
            print(f"❌ {model_file} - Missing (will use untrained model)")
            all_exist = False
    
    return all_exist

def test_webcam():
    """Test if webcam is accessible"""
    print("\n📷 Testing webcam access...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print("✅ Webcam working!")
            cv2.imwrite('outputs/test_webcam.jpg', frame)
            print("✅ Test image saved to outputs/test_webcam.jpg")
        else:
            print("❌ Could not read from webcam")
        cap.release()
    else:
        print("❌ Could not open webcam")

def main():
    print("="*50)
    print("🚗 Driver Safety Monitor - System Test")
    print("="*50)
    
    # Check CUDA
    print(f"\nCUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Check models
    models_exist = test_models_exist()
    
    # Test webcam
    test_webcam()
    
    # Create directories
    os.makedirs('outputs', exist_ok=True)
    
    print("\n" + "="*50)
    if not models_exist:
        print("\n  Some models are missing. You need to train them first!")
        print("\nRun this command to train the distraction detector:")
        print("   python -m src.training.train_driver --mode distraction --epochs 10")
    else:
        print("\n System ready! You can now run:")
        print("   python -m src.inference.safety_monitor")
    print("="*50)

if __name__ == '__main__':
    main()
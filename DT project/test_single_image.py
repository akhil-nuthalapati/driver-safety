import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.models.driver_models import DistractionDetector

def test_single_image(image_path):
    print(f"\n🔍 Testing image: {image_path}")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = DistractionDetector(num_classes=6)
    checkpoint = torch.load('outputs/models/distraction_model_best.pth', 
                           map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # Classes
    classes = ['safe_driving', 'texting', 'talking_phone', 
               'operating_radio', 'drinking', 'reaching_behind']
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load and predict
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    result = {
        'class': classes[predicted.item()],
        'confidence': confidence.item() * 100,
        'all_probs': {classes[i]: probabilities[0][i].item() * 100 
                      for i in range(len(classes))}
    }
    
    print(f"\n✅ Prediction: {result['class']}")
    print(f"   Confidence: {result['confidence']:.2f}%")
    print("\n📊 All probabilities:")
    for cls, prob in sorted(result['all_probs'].items(), 
                           key=lambda x: x[1], reverse=True):
        bar = '█' * int(prob/5)
        print(f"   {cls:20}: {prob:5.2f}% {bar}")
    
    return result

if __name__ == "__main__":
    # Test with webcam snapshot or provide image path
    if len(sys.argv) > 1:
        test_single_image(sys.argv[1])
    else:
        print("Usage: python test_single_image.py <image_path>")
        print("Example: python test_single_image.py outputs/test_webcam.jpg")
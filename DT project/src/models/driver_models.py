import torch
import torch.nn as nn
import torchvision.models as models

class DrowsinessDetector(nn.Module):
    """Specialized model for drowsiness detection"""
    def __init__(self, num_classes=3, pretrained=True):
        super(DrowsinessDetector, self).__init__()
        
        # Use MobileNetV2 for edge deployment
        if pretrained:
            self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        else:
            self.backbone = models.mobilenet_v2(weights=None)
        
        # Modify for num_classes
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Add temporal smoothing
        self.temporal_buffer = []
        self.buffer_size = 10
    
    def forward(self, x):
        return self.backbone(x)
    
    def predict_with_smoothing(self, x):
        """Add temporal smoothing for drowsiness detection"""
        output = self.forward(x)
        prob = torch.softmax(output, dim=1)
        
        # Update buffer
        self.temporal_buffer.append(prob.detach().cpu())
        if len(self.temporal_buffer) > self.buffer_size:
            self.temporal_buffer.pop(0)
        
        # Average over buffer
        if len(self.temporal_buffer) > 0:
            smoothed = torch.mean(torch.stack(self.temporal_buffer), dim=0)
            return smoothed
        return prob

class DistractionDetector(nn.Module):
    """Detect distracted driving behaviors"""
    def __init__(self, num_classes=6, pretrained=True):
        super(DistractionDetector, self).__init__()
        
        # Use ResNet18 for better accuracy
        if pretrained:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.backbone = models.resnet18(weights=None)
            
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class EmotionDetector(nn.Module):
    """Detect driver's emotional state"""
    def __init__(self, num_classes=5, pretrained=True):
        super(EmotionDetector, self).__init__()
        
        # Use ShuffleNet for lightweight deployment
        if pretrained:
            self.backbone = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
        else:
            self.backbone = models.shufflenet_v2_x1_0(weights=None)
            
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class SeatbeltDetector(nn.Module):
    """Binary classifier for seatbelt detection"""
    def __init__(self, num_classes=2, pretrained=True):
        super(SeatbeltDetector, self).__init__()
        
        # Use SqueezeNet for ultra-lightweight
        if pretrained:
            self.backbone = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.DEFAULT)
        else:
            self.backbone = models.squeezenet1_0(weights=None)
            
        self.backbone.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1))
    
    def forward(self, x):
        return self.backbone(x)
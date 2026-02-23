import torch
import torch.nn as nn
import torchvision.models as models


class CNNBackbone(nn.Module):

    def __init__(self, num_classes=10, pretrained=True):
        super(CNNBackbone, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        self.model.to(self.device)

    def forward(self, x):
        return self.model(x)
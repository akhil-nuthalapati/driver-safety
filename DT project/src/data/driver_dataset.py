# src/data/driver_dataset.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

class DriverBehaviorDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='distraction'):
        """
        mode: 'drowsiness', 'distraction', 'emotion', 'seatbelt'
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load class names based on mode
        self.classes = self._get_classes()
        self.image_paths = []
        self.labels = []
        
        # Load dataset (you'll need to download appropriate datasets)
        self._load_dataset()
    
    def _get_classes(self):
        if self.mode == 'drowsiness':
            return ['awake', 'drowsy', 'sleeping']
        elif self.mode == 'distraction':
            return ['safe_driving', 'texting', 'talking_phone', 
                   'operating_radio', 'drinking', 'reaching_behind']
        elif self.mode == 'emotion':
            return ['neutral', 'happy', 'angry', 'surprised', 'frustrated']
        elif self.mode == 'seatbelt':
            return ['seatbelt_on', 'seatbelt_off']
        return []
    
    def _load_dataset(self):
        # TODO: Download and load appropriate datasets
        # Suggested datasets:
        # - Drowsiness: NTHU Drowsy Driver Detection
        # - Distraction: State Farm Distracted Driver Detection
        # - Emotion: AffectNet, FER2013
        # - Seatbelt: Custom dataset
        pass
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# DataLoader factory
def get_driver_dataloaders(mode='distraction', batch_size=32):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # You'll need to split your dataset
    train_dataset = DriverBehaviorDataset(
        root_dir=f'data/driver/{mode}/train',
        transform=transform_train,
        mode=mode
    )
    
    val_dataset = DriverBehaviorDataset(
        root_dir=f'data/driver/{mode}/val',
        transform=transform_val,
        mode=mode
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=4)
    
    return train_loader, val_loader
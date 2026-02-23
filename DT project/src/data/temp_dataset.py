import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np

class TempDriverDataset(Dataset):
    """Temporary dataset using CIFAR-10 as placeholder for driver actions"""
    def __init__(self, mode='distraction', train=True):
        # Load CIFAR-10 as temporary data
        self.cifar = datasets.CIFAR10(
            root='./datasets', 
            train=train, 
            download=True,
            transform=transforms.ToTensor()
        )
        
        # Define classes based on mode
        if mode == 'distraction':
            self.classes = ['safe_driving', 'texting', 'talking_phone', 
                           'operating_radio', 'drinking', 'reaching_behind']
            
            # Map CIFAR-10 classes (0-9) to distraction classes (0-5)
            # This ensures ALL labels are in range 0-5
            self.cifar_to_driver = {
                0: 0,  # airplane -> safe_driving
                1: 0,  # automobile -> safe_driving
                2: 5,  # bird -> reaching_behind
                3: 1,  # cat -> texting
                4: 1,  # deer -> texting
                5: 4,  # dog -> drinking
                6: 2,  # frog -> talking_phone
                7: 3,  # horse -> operating_radio
                8: 0,  # ship -> safe_driving
                9: 2,  # truck -> talking_phone
            }
            
        elif mode == 'drowsiness':
            self.classes = ['awake', 'drowsy', 'sleeping']
            # Map CIFAR-10 to drowsiness classes
            self.cifar_to_driver = {
                0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 0, 7: 1, 8: 0, 9: 2
            }
            
        elif mode == 'emotion':
            self.classes = ['neutral', 'happy', 'angry', 'surprised', 'frustrated']
            # Map CIFAR-10 to emotion classes
            self.cifar_to_driver = {
                0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 0, 6: 1, 7: 2, 8: 3, 9: 4
            }
            
        elif mode == 'seatbelt':
            self.classes = ['seatbelt_off', 'seatbelt_on']
            # Map CIFAR-10 to seatbelt classes
            self.cifar_to_driver = {
                0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1, 8: 0, 9: 1
            }
        
        print(f"\n📊 {mode.upper()} Dataset Info:")
        print(f"   Number of classes: {len(self.classes)}")
        print(f"   Classes: {self.classes}")
        print(f"   Total samples: {len(self.cifar)}")
        
    def __len__(self):
        return len(self.cifar)
    
    def __getitem__(self, idx):
        img, cifar_label = self.cifar[idx]
        
        # Get driver label directly from mapping (already an integer)
        driver_label = self.cifar_to_driver[cifar_label]
        
        # Verify label is valid (for debugging)
        assert 0 <= driver_label < len(self.classes), f"Invalid label {driver_label} for class count {len(self.classes)}"
        
        return img, driver_label

def get_temp_dataloaders(mode='distraction', batch_size=32):
    """Get temporary dataloaders for testing"""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Download CIFAR-10
    print(f"\n📥 Loading CIFAR-10 for {mode} mode...")
    trainset = datasets.CIFAR10(root='./datasets', train=True, 
                                download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./datasets', train=False, 
                               download=True, transform=transform_test)
    
    # Create custom datasets
    train_dataset = TempDriverDataset(mode=mode, train=True)
    test_dataset = TempDriverDataset(mode=mode, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=0, drop_last=True)  # drop_last helps with batch size issues
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=0, drop_last=True)
    
    # Verify label ranges
    print("\n🔍 Verifying label ranges...")
    sample_batch = next(iter(train_loader))
    images, labels = sample_batch
    print(f"   Batch label range: min={labels.min().item()}, max={labels.max().item()}")
    print(f"   Unique labels: {torch.unique(labels).tolist()}")
    print(f"   ✅ All labels valid!")
    
    return train_loader, test_loader
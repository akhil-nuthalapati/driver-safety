import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.driver_models import DistractionDetector
from src.data.temp_dataset import get_temp_dataloaders

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        # DEBUG: Check labels before moving to GPU
        unique_labels = torch.unique(labels)
        if batch_idx == 0:  # Only print for first batch
            print(f"\n🔍 First batch labels: min={labels.min().item()}, max={labels.max().item()}")
            print(f"   Unique labels: {unique_labels.tolist()}")
            print(f"   Label distribution: {torch.bincount(labels)}")
        
        # Check for invalid labels
        if labels.max() >= 6 or labels.min() < 0:
            print(f"\n❌ ERROR: Invalid labels found! Max label: {labels.max()}, Min label: {labels.min()}")
            print(f"   Invalid labels: {(labels >= 6).nonzero().flatten().tolist()}")
            # Fix by clamping
            labels = torch.clamp(labels, 0, 5)
            print(f"   🔧 Fixed labels by clamping to [0,5]")
        
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'Loss': f'{running_loss/(total/len(images)):.3f}', 
                          'Acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss/len(train_loader), 100.*correct/total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating'):
            # Check labels
            if labels.max() >= 6:
                print(f"\n⚠️  Validation - Invalid label found: {labels.max()}")
                labels = torch.clamp(labels, 0, 5)
            
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss/len(val_loader), 100.*correct/total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='distraction',
                       choices=['distraction', 'drowsiness', 'emotion', 'seatbelt'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Create directories
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/logs', exist_ok=True)
    
    # Initialize tensorboard
    writer = SummaryWriter(f'outputs/logs/{args.mode}')
    
    # Get data
    print(f"\n📚 Loading data for {args.mode} mode...")
    from src.data.temp_dataset import get_temp_dataloaders
    train_loader, val_loader = get_temp_dataloaders(
        mode=args.mode, 
        batch_size=args.batch_size
    )
    
    # Initialize model with correct number of classes
    print(f"\n🤖 Initializing {args.mode} detector...")
    num_classes = {
        'distraction': 6,
        'drowsiness': 3,
        'emotion': 5,
        'seatbelt': 2
    }[args.mode]
    
    print(f"   Number of classes: {num_classes}")
    from src.models.driver_models import DistractionDetector
    model = DistractionDetector(num_classes=num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    best_val_acc = 0
    
    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch [{epoch+1}/{args.epochs}]")
        print(f"{'='*50}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, 
                                           criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        print(f"\n📊 Results:")
        print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'outputs/models/{args.mode}_model_best.pth')
            print(f"   ✅ Saved best model (val_acc: {val_acc:.2f}%)")
        
        scheduler.step()
    
    print(f"\n{'='*50}")
    print(f"🎉 Training complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"{'='*50}")
    writer.close()
if __name__ == '__main__':
    main()
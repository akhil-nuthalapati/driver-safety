import torch
import torch.nn as nn
import torch.optim as optim
import time

print("="*60)
print("GPU Training Test")
print("="*60)

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Create a simple model
model = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(128*6*6, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
).to(device)

# Create dummy data
x = torch.randn(64, 3, 32, 32).to(device)
y = torch.randint(0, 10, (64,)).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Test training step
print("\n🔬 Testing training step...")
start_time = time.time()

for i in range(10):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    if i == 0:
        print(f"First forward pass completed")

end_time = time.time()
print(f"✅ Training step successful!")
print(f"   Time for 10 iterations: {(end_time - start_time)*1000:.2f} ms")

if device.type == 'cuda':
    print(f"   GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB")
    print(f"   GPU Memory cached: {torch.cuda.memory_reserved(0) / 1e6:.2f} MB")

print("="*60)
import os
import shutil
import glob

print("🔧 Setting up model files for safety monitor...")
print("="*50)

# Create models directory if it doesn't exist
os.makedirs('outputs/models', exist_ok=True)

# Define model mappings (source -> destination)
model_mappings = {
    'distraction_model_best.pth': 'distraction_model.pth',
    'drowsiness_model_best.pth': 'drowsiness_model.pth',
    'emotion_model_best.pth': 'emotion_model.pth',
    'seatbelt_model_best.pth': 'seatbelt_model.pth'
}

# Copy/rename model files
for source_name, dest_name in model_mappings.items():
    source_path = os.path.join('outputs/models', source_name)
    dest_path = os.path.join('outputs/models', dest_name)
    
    if os.path.exists(source_path):
        # Copy the file (don't remove original)
        shutil.copy2(source_path, dest_path)
        file_size = os.path.getsize(source_path) / (1024*1024)  # Size in MB
        print(f"✅ Copied {source_name} ({file_size:.2f} MB) -> {dest_name}")
    else:
        print(f"⚠️  {source_name} not found - will use untrained model")

# List all model files
print("\n📁 Model files in outputs/models:")
if os.path.exists('outputs/models'):
    for file in sorted(os.listdir('outputs/models')):
        if file.endswith('.pth'):
            size = os.path.getsize(os.path.join('outputs/models', file)) / 1024
            print(f"   - {file} ({size:.1f} KB)")
else:
    print("   No model files found!")

print("\n" + "="*50)
print("🎯 Next step: Run 'python -m src.inference.safety_monitor'")
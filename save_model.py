import torch
from model import ResNet50Classifier

# Load optimized model and params
model = ResNet50Classifier(num_classes=38)
model.load_state_dict(torch.load("model_params.pt", map_location='cpu'))

# Save the complete model
torch.save(model, "model_resnet50.pt")
print("✅ model_resnet50.pt saved successfully.")

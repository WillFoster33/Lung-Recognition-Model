import torch
import torchvision.transforms as transforms
from PIL import Image
from train import LungNet

# Load the trained model
model = LungNet()
model.load_state_dict(torch.load("lung_model.pth"))
model.eval()

# Load and preprocess the test image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_path = "path_to_your_image.jpg"  # Replace with the path to your image
image = Image.open(image_path)
image = transform(image).unsqueeze(0)

# Test the model on the single image
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output.data, 1)

print(f"Predicted class: {predicted.item()}")

# Calculate the accuracy
softmax = torch.nn.Softmax(dim=1)
probabilities = softmax(output)
accuracy = probabilities[0][predicted.item()].item() * 100
print(f"Accuracy: {accuracy:.2f}%")

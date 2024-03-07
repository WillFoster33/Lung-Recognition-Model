import torch
from torchvision import transforms
from PIL import Image
from train import LungNet

def test_model(model_path, test_image_path):
    # Load the trained model
    model = LungNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Define the transforms for test images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess the test image
    image = Image.open(test_image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make predictions
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        switcher = {
            0: "Bacterial Pneumonia",
            1: "Bronchitis",
            2: "Healthy Lungs",
            3: "COVID-19",
            4: "Viral Pneumonia"
        }
        diagnosis = switcher.get(predicted.item(), "Unknown")
        print("Diagnosis:", diagnosis)

    # Calculate accuracy
    softmax = torch.nn.Softmax(dim=1)
    probabilities = softmax(outputs)
    accuracy = probabilities[0][predicted.item()].item() * 100
    print(f"Accuracy: {accuracy:.2f}%")

# Specify the path to the trained model file
model_path = "lung_model.pth"

# Specify the path to the test image
test_image_path = "/Users/aidenlang/Downloads/normalxray.jpeg"

# Test the model
test_model(model_path, test_image_path)

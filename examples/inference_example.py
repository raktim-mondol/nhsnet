import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to path to import nhsnet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the CNN-NHSNet model
from nhsnet.models import CNNNHSNet

def imshow(img):
    """Function to show an image"""
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define the classes for CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Data transformations for inference
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load CIFAR-10 test dataset
    print("Loading CIFAR-10 test dataset...")
    testset = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform)
    testloader = DataLoader(
        testset, batch_size=4, shuffle=True, num_workers=2)
    
    # Get some random test images
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    # Show images
    print("Ground truth labels:")
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
    imshow(torchvision.utils.make_grid(images))
    
    # Load the model
    print("Loading CNN-NHSNet model...")
    model = CNNNHSNet(
        input_channels=3,
        num_classes=10,
        initial_channels=64,
        hybrid_mode='parallel'
    ).to(device)
    
    # Check if a saved model exists
    model_path = os.path.join(os.path.dirname(__file__), '../cnn_nhsnet_model.pth')
    if os.path.exists(model_path):
        print(f"Loading saved model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"No saved model found at {model_path}. Using untrained model.")
    
    # Set model to evaluation mode
    model.eval()
    
    # Make predictions
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Print predictions
    print("Predicted labels:")
    print(' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))
    
    # Evaluate on the entire test set
    print("Evaluating on the entire test set...")
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy on 10000 test images: {100 * correct / total:.2f}%')
    
    # Class-wise accuracy
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                if i < labels.size(0):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
    
    # Print class-wise accuracy
    for i in range(10):
        print(f'Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')

if __name__ == '__main__':
    main() 
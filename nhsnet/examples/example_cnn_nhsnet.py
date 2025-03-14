import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import sys

# Add parent directory to path to import nhsnet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the CNN-NHSNet model
from nhsnet.models import CNNNHSNet

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set parameters
    batch_size = 128
    num_epochs = 5  # Just for demonstration
    learning_rate = 0.001
    
    # Data transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load CIFAR-10 dataset
    print("Downloading and preparing CIFAR-10 dataset...")
    trainset = torchvision.datasets.CIFAR10(
        root='../../data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(
        root='../../data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Create the CNN-NHSNet model
    print("Creating CNN-NHSNet model...")
    model = CNNNHSNet(
        input_channels=3,
        num_classes=10,
        initial_channels=64,
        hebbian_lr=0.005,
        sparsity_ratio=0.3,
        dropout_rate=0.2,
        hybrid_mode='parallel'  # Try 'sequential' or 'adaptive' as well
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass with mixup
            outputs = model(inputs, targets)
            
            # Handle mixup outputs
            if isinstance(outputs, tuple) and len(outputs) == 4:
                outputs, targets_a, targets_b, lam = outputs
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                loss = criterion(outputs, targets)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            
            if isinstance(outputs, tuple) and len(outputs) == 4:
                correct += (lam * predicted.eq(targets_a).float() + 
                           (1 - lam) * predicted.eq(targets_b).float()).sum().item()
            else:
                correct += predicted.eq(targets).sum().item()
            
            if (i + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(trainloader)}], '
                      f'Loss: {running_loss/50:.4f}, Acc: {100.*correct/total:.2f}%')
                running_loss = 0.0
        
        # Test the model
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss/len(testloader):.4f}, '
              f'Test Acc: {100.*correct/total:.2f}%')
    
    print("Training complete!")
    
    # Save the model
    save_path = os.path.join(os.path.dirname(__file__), '../../cnn_nhsnet_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to '{save_path}'")

if __name__ == '__main__':
    main() 
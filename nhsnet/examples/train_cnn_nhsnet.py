import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from nhsnet.models import CNNNHSNet

def get_args():
    parser = argparse.ArgumentParser(description='Train CNN-NHSNet on CIFAR-10/100')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                        help='Dataset to use (default: cifar10)')
    parser.add_argument('--batch-size', type=int, default=128, 
                        help='Input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=200, 
                        help='Number of epochs to train (default: 200)')
    parser.add_argument('--learning-rate', type=float, default=0.001, 
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, 
                        help='Weight decay (default: 5e-4)')
    parser.add_argument('--initial-channels', type=int, default=64, 
                        help='Initial number of channels (default: 64)')
    parser.add_argument('--hebbian-lr', type=float, default=0.005, 
                        help='Hebbian learning rate (default: 0.005)')
    parser.add_argument('--sparsity-ratio', type=float, default=0.3, 
                        help='Sparsity ratio for structured sparse convolutions (default: 0.3)')
    parser.add_argument('--dropout-rate', type=float, default=0.2, 
                        help='Dropout rate (default: 0.2)')
    parser.add_argument('--hybrid-mode', type=str, default='parallel', 
                        choices=['parallel', 'sequential', 'adaptive'],
                        help='Hybrid mode for combining CNN and NHSNet (default: parallel)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed (default: 42)')
    parser.add_argument('--data-dir', type=str, default='./data', 
                        help='Directory for storing input data (default: ./data)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='Save the trained model')
    parser.add_argument('--save-dir', type=str, default='./saved_models',
                        help='Directory to save models (default: ./saved_models)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a checkpoint to resume training from')
    
    return parser.parse_args()

def get_data_loaders(args):
    # Data augmentation and normalization
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
    
    # Choose dataset
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=transform_test)
        num_classes = 10
    else:  # cifar100
        trainset = torchvision.datasets.CIFAR100(
            root=args.data_dir, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(
            root=args.data_dir, train=False, download=True, transform=transform_test)
        num_classes = 100
    
    # Create data loaders
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader, num_classes

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train(model, trainloader, optimizer, criterion, device, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass with mixup
        outputs = model(inputs, targets)
        
        # Handle mixup outputs
        if isinstance(outputs, tuple) and len(outputs) == 4:
            outputs, targets_a, targets_b, lam = outputs
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            # For accuracy calculation, use the original targets
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a).float() + 
                       (1 - lam) * predicted.eq(targets_b).float()).sum().item()
        else:
            loss = criterion(outputs, targets)
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # Print progress
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(trainloader):
            print(f'Epoch: {epoch} | Batch: {batch_idx+1}/{len(trainloader)} | '
                  f'Loss: {train_loss/(batch_idx+1):.3f} | '
                  f'Acc: {100.*correct/total:.3f}%')
    
    return train_loss / len(trainloader), 100. * correct / total

def test(model, testloader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return test_loss / len(testloader), 100. * correct / total

def main():
    args = get_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Get data loaders
    trainloader, testloader, num_classes = get_data_loaders(args)
    
    # Create model
    model = CNNNHSNet(
        input_channels=3,
        num_classes=num_classes,
        initial_channels=args.initial_channels,
        hebbian_lr=args.hebbian_lr,
        sparsity_ratio=args.sparsity_ratio,
        dropout_rate=args.dropout_rate,
        hybrid_mode=args.hybrid_mode
    ).to(device)
    
    # Print model summary
    print(f'Model: CNN-NHSNet with {args.hybrid_mode} mode')
    print(f'Initial channels: {args.initial_channels}')
    print(f'Hebbian learning rate: {args.hebbian_lr}')
    print(f'Sparsity ratio: {args.sparsity_ratio}')
    print(f'Dropout rate: {args.dropout_rate}')
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f'Loading checkpoint {args.resume}')
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            print(f'Loaded checkpoint (epoch {start_epoch})')
        else:
            print(f'No checkpoint found at {args.resume}')
    
    # Create save directory if needed
    if args.save_model and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Training loop
    print(f'Starting training for {args.epochs} epochs...')
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        # Train and test
        train_loss, train_acc = train(model, trainloader, optimizer, criterion, device, epoch)
        test_loss, test_acc = test(model, testloader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{args.epochs} | '
              f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}% | '
              f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}% | '
              f'Time: {epoch_time:.2f}s')
        
        # Save model if it's the best so far
        if args.save_model and test_acc > best_acc:
            best_acc = test_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc': best_acc
            }
            torch.save(checkpoint, os.path.join(args.save_dir, f'cnn_nhsnet_{args.hybrid_mode}_best.pth'))
            print(f'Saved best model with accuracy {best_acc:.3f}%')
    
    print(f'Training complete! Best accuracy: {best_acc:.3f}%')

if __name__ == '__main__':
    main() 
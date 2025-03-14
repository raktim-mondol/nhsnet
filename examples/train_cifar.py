import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random

from nhsnet.models.nhsnet import NHSNet
from nhsnet.utils.pruning import AdaptiveSynapticPruning

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_transforms():
    """Get data augmentation and normalization transforms"""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandAugment(num_ops=2, magnitude=9),  # Add RandAugment
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    return train_transform, test_transform

def get_data_loaders(batch_size, num_workers=2):
    """Create data loaders for CIFAR-10"""
    train_transform, test_transform = get_transforms()
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, pruning, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Training')
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with mixup
        outputs = model(inputs, targets)
        
        # Handle mixup outputs
        if isinstance(outputs, tuple) and len(outputs) == 4:
            outputs, targets_a, targets_b, lam = outputs
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            # For accuracy calculation, use the dominant target
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a).float() + (1 - lam) * predicted.eq(targets_b).float()).sum().item()
        else:
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
            
        if pruning is not None:
            pruning.step()
            pruning.apply_masks()
        
        total_loss += loss.item()
        
        progress_bar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total,
            'lr': optimizer.param_groups[0]['lr']
        })
    
    return total_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    
    return avg_loss, accuracy

def train(args):
    """Main training function"""
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = NHSNet(
        input_channels=3,
        num_classes=10,
        initial_channels=args.initial_channels,
        num_blocks=[3, 4, 6, 3],  # ResNet-like architecture
        hebbian_lr=args.hebbian_lr,
        sparsity_ratio=args.sparsity_ratio,
        dropout_rate=args.dropout,
        use_se=True  # Enable Squeeze-Excitation
    ).to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Add label smoothing
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(args.batch_size)
    
    # Calculate steps per epoch
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    
    # Learning rate scheduler with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        div_factor=25,  # initial_lr = max_lr/25
        final_div_factor=1000  # min_lr = initial_lr/1000
    )
    
    # Initialize pruning
    pruning = AdaptiveSynapticPruning(
        model,
        pruning_ratio=args.sparsity_ratio,
        pruning_interval=100
    )
    
    # Training loop
    best_acc = 0
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}:')
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, pruning, device
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
        
        # Save best model
        if test_acc > best_acc:
            print(f'New best model saved with accuracy: {test_acc:.2f}%')
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc
            }
            torch.save(checkpoint, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NHS-Net on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hebbian-lr', type=float, default=0.005, help='hebbian learning rate')
    parser.add_argument('--sparsity-ratio', type=float, default=0.3, help='sparsity ratio')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--initial-channels', type=int, default=64, help='initial channels')
    parser.add_argument('--device', type=str, default='cuda', help='device to use')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='directory to save models')
    
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    train(args)
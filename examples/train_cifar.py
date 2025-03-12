import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse

from nhsnet.models import NHSNet
from nhsnet.utils.pruning import AdaptiveSynapticPruning

def parse_args():
    parser = argparse.ArgumentParser(description='Train NHS-Net on CIFAR-100')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--hebbian-lr', type=float, default=0.01,
                        help='hebbian learning rate')
    parser.add_argument('--sparsity-ratio', type=float, default=0.5,
                        help='sparsity ratio for structured sparse layers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to train on (cuda or cpu)')
    return parser.parse_args()

def train(args):
    print(f"Training NHS-Net with parameters:")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Hebbian learning rate: {args.hebbian_lr}")
    print(f"Sparsity ratio: {args.sparsity_ratio}")
    print(f"Device: {args.device}")

    # Data loading
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR100(
        root='./data', 
        train=True,
        download=True, 
        transform=transform_train
    )
    trainloader = DataLoader(
        trainset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=2
    )

    testset = datasets.CIFAR100(
        root='./data', 
        train=False,
        download=True, 
        transform=transform_test
    )
    testloader = DataLoader(
        testset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=2
    )

    # Model setup
    model = NHSNet(
        input_channels=3,
        num_classes=100,
        hebbian_lr=args.hebbian_lr,
        sparsity_ratio=args.sparsity_ratio
    ).to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Initialize pruning
    pruning = AdaptiveSynapticPruning(
        model,
        pruning_ratio=0.1,
        pruning_interval=100
    )

    best_acc = 0.0
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        with tqdm(trainloader, desc=f'Epoch {epoch+1}/{args.epochs}') as pbar:
            for inputs, targets in pbar:
                inputs = inputs.to(args.device)
                targets = targets.to(args.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                # Apply pruning
                pruning.step()
                pruning.apply_masks()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar.set_postfix({
                    'loss': running_loss/len(pbar),
                    'acc': 100.*correct/total
                })

        # Validation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in testloader:
                inputs = inputs.to(args.device)
                targets = targets.to(args.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100.*correct/total
        
        # Print metrics
        print(f'\nEpoch: {epoch+1}')
        print(f'Train Loss: {running_loss/len(trainloader):.3f}')
        print(f'Test Loss: {test_loss/len(testloader):.3f}')
        print(f'Test Accuracy: {test_acc:.2f}%')

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, 'best_model.pth')
            print(f'New best model saved with accuracy: {test_acc:.2f}%')

        scheduler.step()

    print(f"\nTraining completed!")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Best model saved as 'best_model.pth'")

if __name__ == "__main__":
    args = parse_args()
    train(args)
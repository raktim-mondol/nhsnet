import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import wandb

from nhsnet.models import NHSNet
from nhsnet.utils.pruning import AdaptiveSynapticPruning

def train(
    epochs=100,
    batch_size=128,
    learning_rate=0.001,
    hebbian_lr=0.01,
    sparsity_ratio=0.5,
    device="cuda"
):
    # Initialize wandb
    wandb.init(project="nhsnet", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hebbian_lr": hebbian_lr,
        "sparsity_ratio": sparsity_ratio
    })

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
        batch_size=batch_size,
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
        batch_size=batch_size,
        shuffle=False, 
        num_workers=2
    )

    # Model setup
    model = NHSNet(
        input_channels=3,
        num_classes=100,
        hebbian_lr=hebbian_lr,
        sparsity_ratio=sparsity_ratio
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Initialize pruning
    pruning = AdaptiveSynapticPruning(
        model,
        pruning_ratio=0.1,
        pruning_interval=100
    )

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        with tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}') as pbar:
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)

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
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100.*correct/total
        
        # Log metrics
        wandb.log({
            "train_loss": running_loss/len(trainloader),
            "test_loss": test_loss/len(testloader),
            "test_acc": test_acc
        })

        scheduler.step()

    wandb.finish()

if __name__ == "__main__":
    train()
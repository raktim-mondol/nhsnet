# CNN-NHSNet: Hybrid Neural Network with Neuroplasticity

This repository contains a hybrid neural network architecture that combines traditional Convolutional Neural Networks (CNNs) with the Neuroplasticity features of NHSNet (Neural Hebbian-Sparse Network).

## Features

- **Hybrid Architecture**: Combines traditional CNN components with neuroplasticity features
- **Multiple Hybrid Modes**:
  - `parallel`: Processes inputs through both CNN and NHSNet paths simultaneously
  - `sequential`: Alternates between CNN blocks and NHSNet blocks
  - `adaptive`: Uses CNN blocks in early stages and NHSNet blocks in later stages
- **Dynamic Neurogenesis**: Automatically adds new neurons during training based on activation patterns
- **Hebbian Learning**: Implements biologically-inspired learning mechanisms
- **Structured Sparsity**: Maintains sparse connectivity for efficiency
- **Robust Device Handling**: Ensures all components are on the same device during training

## Requirements

- Python 3.6+
- PyTorch 1.7+
- torchvision
- matplotlib (for visualization)
- numpy

## Example Scripts

### Training Example

The `example_cnn_nhsnet.py` script demonstrates how to train the CNN-NHSNet model on the CIFAR-10 dataset:

```bash
python example_cnn_nhsnet.py
```

This script:
1. Downloads and prepares the CIFAR-10 dataset
2. Creates a CNN-NHSNet model with the 'parallel' hybrid mode
3. Trains the model for 5 epochs (for demonstration)
4. Evaluates the model on the test set
5. Saves the trained model to 'cnn_nhsnet_model.pth'

### Inference Example

The `inference_example.py` script demonstrates how to use the trained model for inference:

```bash
python inference_example.py
```

This script:
1. Loads the CIFAR-10 test dataset
2. Displays some sample images with their ground truth labels
3. Loads the CNN-NHSNet model (either pre-trained or untrained)
4. Makes predictions on the sample images
5. Evaluates the model on the entire test set
6. Displays class-wise accuracy

## Advanced Training

For more advanced training options, use the `nhsnet/examples/train_cnn_nhsnet.py` script:

```bash
python nhsnet/examples/train_cnn_nhsnet.py --epochs 200 --batch-size 128 --learning-rate 0.001 --initial-channels 64 --hybrid-mode parallel --save-model
```

Available options:
- `--dataset`: Choose between 'cifar10' and 'cifar100'
- `--batch-size`: Input batch size for training
- `--epochs`: Number of epochs to train
- `--learning-rate`: Learning rate
- `--weight-decay`: Weight decay for regularization
- `--initial-channels`: Initial number of channels
- `--hebbian-lr`: Hebbian learning rate
- `--sparsity-ratio`: Sparsity ratio for structured sparse convolutions
- `--dropout-rate`: Dropout rate
- `--hybrid-mode`: Hybrid mode ('parallel', 'sequential', or 'adaptive')
- `--save-model`: Flag to save the trained model
- `--save-dir`: Directory to save models
- `--resume`: Path to a checkpoint to resume training from

## Model Architecture

The CNN-NHSNet architecture consists of:

1. **Stem**: Initial convolutional layers for feature extraction
2. **Stages**: Multiple stages with increasing channel dimensions
3. **Hybrid Blocks**: Blocks that combine CNN and NHSNet features
4. **Neurogenesis Module**: Dynamically adds neurons during training
5. **Classification Head**: Final layers for classification

## Performance

The hybrid approach offers several benefits:
- Improved stability during training
- Better feature extraction capabilities
- Adaptive learning through neuroplasticity
- Flexible architecture that can be tailored to specific tasks

## Citation

If you use this code in your research, please cite:

```
@article{nhsnet2025,
  title={CNN-NHSNet: A Hybrid Neural Network with Neuroplasticity},
  author={Mondol, Raktim},
  journal={arXiv preprint},
  year={2025}
}
```
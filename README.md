# NHS-Net: Neural Hebbian System Network

A PyTorch implementation of NHS-Net, featuring Hebbian learning, structured sparsity, and dynamic neurogenesis.

## Features

- Hebbian Convolutional Layer with local learning rules
- Structured Sparse Convolution for efficient computation
- Hodgkin-Huxley Gating mechanism
- Dynamic Neurogenesis Module
- Adaptive Synaptic Pruning

## Installation

### From Source

1. Clone the repository:
```bash
git clone https://github.com/raktim-mondol/nhsnet.git
cd nhsnet
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate
```

3. Install in development mode:
```bash
pip install -e .
```

### Requirements

- Python >= 3.7
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- numpy >= 1.19.2
- scipy >= 1.7.0
- tqdm >= 4.62.0

## Usage

### Training on CIFAR-10

```python
from nhsnet import NHSNet
import torch

# Create model
model = NHSNet(
    in_channels=3,
    num_classes=10,
    hidden_channels=[64, 128, 256],
    hebbian_lr=0.01
)

# Train the model
python examples/train_cifar.py --epochs 100 --batch-size 128 --lr 0.001
```

## Development

### Setup Development Environment

1. Install development dependencies:
```bash
pip install -r requirements.txt
```

2. Install pre-commit hooks:
```bash
pre-commit install
```

### Code Quality

We use several tools to maintain code quality:

- `black` for code formatting
- `isort` for import sorting
- `flake8` for style guide enforcement
- `mypy` for static type checking

Run all checks:
```bash
# Format code
black .
isort .

# Check code
flake8 .
mypy .
```

### Testing

Run tests with coverage:
```bash
pytest --cov=nhsnet tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use NHS-Net in your research, please cite:

```bibtex
@article{nhsnet2024,
  title={NHS-Net: Neural Hebbian System Network},
  author={Mondol, Raktim},
  year={2024}
}
```
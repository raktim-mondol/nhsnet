# NHS-Net: Neurogenesis-Hebbian Sparse Network

NHS-Net is a novel deep learning architecture that integrates biological principles including:
- Hebbian Learning
- Neurogenesis
- Synaptic Pruning
- Sparse Neural Circuitry
- Hodgkin-Huxley Gating

## Features
- Dynamic architecture adaptation through neurogenesis
- Energy-efficient sparse connectivity
- Biologically-inspired Hebbian learning
- Adaptive synaptic pruning
- Voltage-gated activations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/raktim-mondol/nhsnet.git
cd nhsnet
```

2. Install dependencies and the package in development mode:
```bash
pip install -r requirements.txt
pip install -e .
```

The `-e` flag installs the package in "editable" mode, which means you can modify the source code without reinstalling.

## Usage

### Basic Model Usage
```python
from nhsnet.models import NHSNet

model = NHSNet(
    input_channels=3,
    num_classes=1000,
    hebbian_lr=0.01,
    sparsity_ratio=0.5
)
```

### Training on CIFAR-100

The repository includes a complete training script for CIFAR-100. To train the model:

```bash
# From the root directory:
python examples/train_cifar.py

# Or from the examples directory:
cd examples
python train_cifar.py
```

You can customize the training parameters:

```bash
python examples/train_cifar.py \
    --epochs 200 \
    --batch-size 128 \
    --learning-rate 0.001 \
    --hebbian-lr 0.01 \
    --sparsity-ratio 0.5 \
    --device cuda
```

Available parameters:
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 128)
- `--learning-rate`: Learning rate for optimizer (default: 0.001)
- `--hebbian-lr`: Learning rate for Hebbian updates (default: 0.01)
- `--sparsity-ratio`: Sparsity ratio for structured sparse layers (default: 0.5)
- `--device`: Device to train on ('cuda' or 'cpu', default: 'cuda')

The training script includes:
- Data augmentation for CIFAR-100
- Learning rate scheduling
- Adaptive synaptic pruning
- Progress bars with tqdm
- Best model checkpointing
- Detailed training metrics

### Model Architecture
NHS-Net combines several biologically-inspired mechanisms:

1. **Hebbian Learning**: Implements "neurons that fire together, wire together" principle
2. **Neurogenesis**: Dynamically adds neurons during training
3. **Synaptic Pruning**: Removes weak connections to improve efficiency
4. **Sparse Connectivity**: Maintains biologically-plausible sparse neural circuits
5. **Hodgkin-Huxley Gating**: Uses voltage-like gating mechanisms

## Project Structure
```
nhsnet/
├── nhsnet/
│   ├── __init__.py
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── hebbian_conv.py
│   │   ├── structured_sparse.py
│   │   ├── hh_gating.py
│   │   └── dynamic_neurogenesis.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── nhsnet.py
│   └── utils/
│       ├── __init__.py
│       └── pruning.py
├── examples/
│   └── train_cifar.py
├── setup.py
├── requirements.txt
└── README.md
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
MIT

## Citation
If you use NHS-Net in your research, please cite:
```bibtex
@article{nhsnet2024,
    title={NHS-Net: A Biologically-Inspired Neural Architecture},
    author={Mondol, Raktim},
    year={2024}
}
```
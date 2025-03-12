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
```bash
pip install -r requirements.txt
```

## Usage
```python
from nhsnet.models import NHSNet

model = NHSNet(
    input_channels=3,
    num_classes=1000,
    hebbian_lr=0.01,
    sparsity_ratio=0.5
)
```

## Architecture
NHS-Net combines several biologically-inspired mechanisms:

1. **Hebbian Learning**: Implements "neurons that fire together, wire together" principle
2. **Neurogenesis**: Dynamically adds neurons during training
3. **Synaptic Pruning**: Removes weak connections to improve efficiency
4. **Sparse Connectivity**: Maintains biologically-plausible sparse neural circuits
5. **Hodgkin-Huxley Gating**: Uses voltage-like gating mechanisms

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
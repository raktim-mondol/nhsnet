import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.hebbian_conv import HebbianConv2d
from ..layers.structured_sparse import StructuredSparseConv2d
from ..layers.hh_gating import HodgkinHuxleyGating
from ..layers.dynamic_neurogenesis import DynamicNeurogenesisModule

class NHSNetBlock(nn.Module):
    """Basic building block for NHS-Net"""
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 hebbian_lr=0.01,
                 sparsity_ratio=0.5):
        super().__init__()
        
        self.conv1 = HebbianConv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            padding=1,
            hebbian_lr=hebbian_lr
        )
        
        self.sparse_conv = StructuredSparseConv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            sparsity_ratio=sparsity_ratio
        )
        
        self.gating = HodgkinHuxleyGating(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.sparse_conv(x)
        x = self.gating(x)
        x = self.bn(x)
        return F.relu(x)

class NHSNet(nn.Module):
    """Complete NHS-Net architecture"""
    def __init__(self,
                 input_channels=3,
                 num_classes=1000,
                 initial_channels=64,
                 num_blocks=4,
                 hebbian_lr=0.01,
                 sparsity_ratio=0.5):
        super().__init__()
        
        self.input_conv = nn.Conv2d(
            input_channels, 
            initial_channels, 
            kernel_size=7, 
            stride=2, 
            padding=3
        )
        
        self.blocks = nn.ModuleList()
        current_channels = initial_channels
        
        for i in range(num_blocks):
            out_channels = current_channels * 2
            self.blocks.append(
                NHSNetBlock(
                    current_channels,
                    out_channels,
                    hebbian_lr=hebbian_lr,
                    sparsity_ratio=sparsity_ratio
                )
            )
            current_channels = out_channels
            
        self.neurogenesis = DynamicNeurogenesisModule(
            initial_neurons=current_channels,
            max_neurons=current_channels * 2
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(current_channels, num_classes)
        
    def forward(self, x):
        x = self.input_conv(x)
        
        for block in self.blocks:
            x = block(x)
            
            # Apply neurogenesis if needed
            if self.training:
                activation_patterns = x.detach()
                mean_activation, under_activated = \
                    self.neurogenesis.compute_activation_statistics(activation_patterns)
                
                if under_activated.any():
                    block.conv1 = self.neurogenesis.expand_layer(
                        block.conv1,
                        activation_patterns
                    )
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
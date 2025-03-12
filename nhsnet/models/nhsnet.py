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
        
        self.hebbian_lr = hebbian_lr  # Store hebbian_lr at block level
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
        
    def update_channels(self, new_channels):
        """Update all layers to match new channel dimensions"""
        device = self.sparse_conv.weight.device
        
        # Update sparse conv
        self.sparse_conv = StructuredSparseConv2d(
            new_channels,
            new_channels,
            kernel_size=3,
            padding=1,
            sparsity_ratio=self.sparse_conv.sparsity_ratio
        ).to(device)
        
        # Update gating
        self.gating = HodgkinHuxleyGating(new_channels).to(device)
        
        # Update batch norm
        self.bn = nn.BatchNorm2d(new_channels).to(device)
        
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
        
    def to(self, device):
        """Override to method to ensure proper device placement"""
        super().to(device)
        # Ensure gating buffers are on the correct device
        for block in self.blocks:
            block.gating.reset_state()  # This will create new buffers on the correct device
        return self
        
    def forward(self, x):
        x = self.input_conv(x)
        
        for i, block in enumerate(self.blocks):
            # Apply neurogenesis if needed
            if self.training:
                activation_patterns = x.detach()
                mean_activation, under_activated = \
                    self.neurogenesis.compute_activation_statistics(activation_patterns)
                
                if under_activated.any():
                    # Expand the current block's conv1 layer
                    new_conv1 = self.neurogenesis.expand_layer(
                        block.conv1,
                        activation_patterns
                    )
                    
                    if new_conv1 is not block.conv1:  # If layer was expanded
                        device = x.device  # Use input tensor's device
                        block.conv1 = new_conv1.to(device)
                        block.update_channels(new_conv1.out_channels)
                        
                        # Update next block's input channels if it exists
                        if i < len(self.blocks) - 1:
                            next_block = self.blocks[i + 1]
                            # Create new HebbianConv2d with stored hebbian_lr
                            next_block.conv1 = HebbianConv2d(
                                new_conv1.out_channels,
                                next_block.conv1.out_channels,
                                kernel_size=3,
                                padding=1,
                                hebbian_lr=next_block.hebbian_lr  # Use stored hebbian_lr
                            ).to(device)
                        else:
                            # Update classifier input features for the last block
                            old_weight = self.classifier.weight.data
                            old_bias = self.classifier.bias.data
                            self.classifier = nn.Linear(
                                new_conv1.out_channels,
                                self.classifier.out_features
                            ).to(device)
                            
                            # Initialize with zeros for new features
                            self.classifier.weight.data[:, :old_weight.size(1)] = old_weight
                            self.classifier.bias.data = old_bias
            
            x = block(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.hebbian_conv import HebbianConv2d
from ..layers.structured_sparse import StructuredSparseConv2d
from ..layers.hh_gating import HodgkinHuxleyGating
from ..layers.dynamic_neurogenesis import DynamicNeurogenesisModule

class NHSNetBlock(nn.Module):
    """Enhanced building block for NHS-Net with residual connections"""
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 hebbian_lr=0.01,
                 sparsity_ratio=0.5,
                 stride=1):
        super().__init__()
        
        self.hebbian_lr = hebbian_lr
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Main path
        self.conv1 = HebbianConv2d(
            in_channels, 
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            hebbian_lr=hebbian_lr
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.sparse_conv = StructuredSparseConv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            sparsity_ratio=sparsity_ratio
        )
        
        # Gating mechanism
        self.gating = HodgkinHuxleyGating(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
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
        
        # Update shortcut if needed
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_channels, new_channels, kernel_size=1, stride=self.shortcut[0].stride),
                nn.BatchNorm2d(new_channels)
            ).to(device)
        
        self.out_channels = new_channels
        
    def forward(self, x):
        identity = x
        
        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.sparse_conv(out)
        out = self.gating(out)
        
        # Add shortcut connection
        out += self.shortcut(identity)
        out = F.relu(out)
        
        # Apply dropout
        out = self.dropout(out)
        
        return out

class NHSNet(nn.Module):
    """Enhanced NHS-Net architecture"""
    def __init__(self,
                 input_channels=3,
                 num_classes=1000,
                 initial_channels=64,
                 num_blocks=[2, 2, 2, 2],
                 hebbian_lr=0.01,
                 sparsity_ratio=0.5):
        super().__init__()
        
        self.initial_channels = initial_channels
        
        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, initial_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Build network stages
        self.stages = nn.ModuleList()
        current_channels = initial_channels
        
        for i, num_blocks_in_stage in enumerate(num_blocks):
            stage = []
            for j in range(num_blocks_in_stage):
                stride = 2 if j == 0 and i > 0 else 1
                stage.append(
                    NHSNetBlock(
                        current_channels,
                        current_channels * 2 if j == 0 else current_channels * 2,
                        hebbian_lr=hebbian_lr,
                        sparsity_ratio=sparsity_ratio,
                        stride=stride
                    )
                )
                if j == 0:
                    current_channels *= 2
            self.stages.append(nn.Sequential(*stage))
            
        self.neurogenesis = DynamicNeurogenesisModule(
            initial_neurons=current_channels,
            max_neurons=current_channels * 2
        )
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(current_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def to(self, device):
        """Override to method to ensure proper device placement"""
        super().to(device)
        # Ensure gating buffers are on the correct device
        for stage in self.stages:
            for block in stage:
                if hasattr(block, 'gating'):
                    block.gating.reset_state()
        return self
        
    def forward(self, x):
        x = self.stem(x)
        
        # Process through stages
        for stage_idx, stage in enumerate(self.stages):
            for block_idx, block in enumerate(stage):
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
                            device = x.device
                            block.conv1 = new_conv1.to(device)
                            block.update_channels(new_conv1.out_channels)
                            
                            # Update next block if it exists
                            if block_idx < len(stage) - 1:
                                next_block = stage[block_idx + 1]
                                next_block.conv1 = HebbianConv2d(
                                    new_conv1.out_channels,
                                    next_block.out_channels,
                                    kernel_size=3,
                                    padding=1,
                                    hebbian_lr=next_block.hebbian_lr
                                ).to(device)
                                next_block.in_channels = new_conv1.out_channels
                
                x = block(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
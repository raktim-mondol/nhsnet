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
        self.stride = stride
        
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
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Gating mechanism
        self.gating = HodgkinHuxleyGating(out_channels)
        
        # Shortcut connection
        self._make_shortcut()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def _make_shortcut(self):
        """Create or update shortcut connection"""
        if self.stride != 1 or self.in_channels != self.out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, 
                         kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(self.out_channels)
            )
        else:
            self.shortcut = nn.Identity()

class NHSNet(nn.Module):
    """Enhanced NHS-Net architecture"""
    def __init__(self,
                 input_channels=3,  # Changed from 256 to 3 for CIFAR-10
                 num_classes=10,
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
            nn.ReLU()
        )
        
        # Build network stages
        self.stages = nn.ModuleList()
        current_channels = initial_channels
        
        for i, num_blocks_in_stage in enumerate(num_blocks):
            stage = []
            for j in range(num_blocks_in_stage):
                # Only double channels at the beginning of each stage (except first)
                out_channels = current_channels * 2 if j == 0 and i > 0 else current_channels
                stride = 2 if j == 0 and i > 0 else 1
                
                block = NHSNetBlock(
                    current_channels,
                    out_channels,
                    hebbian_lr=hebbian_lr,
                    sparsity_ratio=sparsity_ratio,
                    stride=stride
                )
                stage.append(block)
                current_channels = out_channels
                
            self.stages.append(nn.Sequential(*stage))
        
        self.neurogenesis = DynamicNeurogenesisModule(
            initial_neurons=current_channels,
            max_neurons=current_channels * 2
        )
        
        # Global pooling and classification
        self.avgpool = nn.AdaptiveAvgPool2d(1)
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
            if isinstance(m, (nn.Conv2d, HebbianConv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
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
    
    def _update_block_channels(self, block, new_channels, next_block=None):
        """Helper method to update block channels and maintain consistency"""
        # Update current block's conv1
        device = next(block.parameters()).device
        
        # Update conv1 and its output channels
        block.conv1 = HebbianConv2d(
            block.in_channels,
            new_channels,
            kernel_size=3,
            stride=block.stride,
            padding=1,
            hebbian_lr=block.hebbian_lr
        ).to(device)
        
        # Update batch norm layers
        block.bn1 = nn.BatchNorm2d(new_channels).to(device)
        block.bn2 = nn.BatchNorm2d(new_channels).to(device)
        
        # Update sparse conv to match new channels
        block.sparse_conv = StructuredSparseConv2d(
            new_channels,
            new_channels,
            kernel_size=3,
            padding=1,
            sparsity_ratio=block.sparse_conv.sparsity_ratio
        ).to(device)
        
        # Update gating mechanism
        block.gating = HodgkinHuxleyGating(new_channels).to(device)
        
        # Update output channels
        block.out_channels = new_channels
        
        # Update shortcut connection
        if block.stride != 1 or block.in_channels != new_channels:
            block.shortcut = nn.Sequential(
                nn.Conv2d(block.in_channels, new_channels, 
                         kernel_size=1, stride=block.stride, bias=False),
                nn.BatchNorm2d(new_channels)
            ).to(device)
        
        # Update next block's input channels if it exists
        if next_block is not None:
            next_block.in_channels = new_channels
            next_block._make_shortcut()
            next_block = next_block.to(device)
    
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
                            new_out_channels = new_conv1.out_channels
                            
                            # Get next block if it exists
                            next_block = None
                            if block_idx < len(stage) - 1:
                                next_block = stage[block_idx + 1]
                            elif stage_idx < len(self.stages) - 1:
                                next_block = self.stages[stage_idx + 1][0]
                            
                            # Update channels throughout the block and next block
                            self._update_block_channels(
                                block,
                                new_out_channels,
                                next_block
                            )
                
                # Forward pass through the block
                identity = x
                
                # Main path with channel checks
                out = block.conv1(x)  # This should now have matching channels
                out = block.bn1(out)
                out = F.relu(out)
                
                out = block.sparse_conv(out)
                out = block.bn2(out)
                out = block.gating(out)
                
                # Add shortcut connection
                out = out + block.shortcut(identity)
                out = F.relu(out)
                
                # Apply dropout
                out = block.dropout(out)
                
                x = out
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
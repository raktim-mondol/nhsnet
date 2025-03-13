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
        
    def update_channels(self, new_out_channels):
        """Update all layers to match new channel dimensions"""
        if new_out_channels == self.out_channels:
            return
            
        device = next(self.parameters()).device
        old_out_channels = self.out_channels
        
        # Update conv1
        self.conv1 = HebbianConv2d(
            self.in_channels,
            new_out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            hebbian_lr=self.hebbian_lr
        ).to(device)
        
        # Update batch norm layers
        self.bn1 = nn.BatchNorm2d(new_out_channels).to(device)
        self.bn2 = nn.BatchNorm2d(new_out_channels).to(device)
        
        # Update sparse conv
        self.sparse_conv = StructuredSparseConv2d(
            new_out_channels,
            new_out_channels,
            kernel_size=3,
            padding=1,
            sparsity_ratio=self.sparse_conv.sparsity_ratio
        ).to(device)
        
        # Update gating
        self.gating = HodgkinHuxleyGating(new_out_channels).to(device)
        
        # Update output channels
        self.out_channels = new_out_channels
        
        # Update shortcut
        self._make_shortcut()
        self.shortcut = self.shortcut.to(device)
        
    def forward(self, x):
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.sparse_conv(out)
        out = self.bn2(out)
        out = self.gating(out)
        
        # Add shortcut connection
        out = out + self.shortcut(identity)
        out = F.relu(out)
        
        # Apply dropout
        out = self.dropout(out)
        
        return out

class NHSNet(nn.Module):
    """Enhanced NHS-Net architecture"""
    def __init__(self,
                 input_channels=3,
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
        # Update current block
        block.update_channels(new_channels)
        
        # Update next block's input channels if it exists
        if next_block is not None:
            next_block.in_channels = new_channels
            next_block._make_shortcut()  # Update shortcut connection
            next_block = next_block.to(block.conv1.weight.device)
    
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
                            new_conv1 = new_conv1.to(device)
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
                
                x = block(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
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
                 stride=1,
                 residual_scale=0.1):
        super().__init__()
        
        self.hebbian_lr = hebbian_lr
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.residual_scale = residual_scale
        
        # Main path with channel normalization
        self.conv1 = HebbianConv2d(
            in_channels, 
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            hebbian_lr=hebbian_lr,
            bias=False  # Remove bias when using BatchNorm
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Structured sparse convolution
        self.sparse_conv = StructuredSparseConv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            sparsity_ratio=sparsity_ratio,
            bias=False  # Remove bias when using BatchNorm
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Gating mechanism
        self.gating = HodgkinHuxleyGating(out_channels)
        
        # Shortcut connection
        self._make_shortcut()
        
        # Reduced dropout for better stability
        self.dropout = nn.Dropout(0.1)
        
    def _make_shortcut(self):
        """Create or update shortcut connection"""
        if self.stride != 1 or self.in_channels != self.out_channels:
            # Get device from existing parameters
            device = next(self.parameters()).device
            
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    self.out_channels,
                    kernel_size=1,
                    stride=self.stride,
                    bias=False  # Remove bias when using BatchNorm
                ).to(device),
                nn.BatchNorm2d(self.out_channels).to(device)
            )
        else:
            self.shortcut = nn.Identity()
            
    def _apply_layer_norm(self, x):
        """Apply instance normalization for better stability"""
        # Normalize each channel independently
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + 1e-5)
        
    def reset_parameters(self):
        """Reset all parameters of the block"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, HebbianConv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _update_channels(self, new_channels):
        """Update the block's channel dimensions"""
        device = next(self.parameters()).device
        
        # Store original channels
        original_in = self.in_channels
        
        # Update output channels
        self.out_channels = new_channels
        
        # Update main path
        self.conv1 = HebbianConv2d(
            original_in, new_channels,
            kernel_size=3, stride=self.stride,
            padding=1, hebbian_lr=self.hebbian_lr,
            bias=False
        ).to(device)
        
        self.bn1 = nn.BatchNorm2d(new_channels).to(device)
        
        self.sparse_conv = StructuredSparseConv2d(
            new_channels, new_channels,
            kernel_size=3, padding=1,
            sparsity_ratio=self.sparse_conv.sparsity_ratio,
            bias=False
        ).to(device)
        
        self.bn2 = nn.BatchNorm2d(new_channels).to(device)
        self.gating = HodgkinHuxleyGating(new_channels).to(device)
        
        # Update shortcut
        self._make_shortcut()
        
        # Reset parameters
        self.reset_parameters()
        return self
        
    def to(self, device):
        """Override to method to ensure proper device placement"""
        super().to(device)
        # Ensure all components are on the correct device
        self.conv1 = self.conv1.to(device)
        self.bn1 = self.bn1.to(device)
        self.sparse_conv = self.sparse_conv.to(device)
        self.bn2 = self.bn2.to(device)
        self.gating = self.gating.to(device)
        self.shortcut = self.shortcut.to(device)
        return self

class NHSNet(nn.Module):
    """Enhanced NHS-Net architecture with stable channel management"""
    def __init__(self,
                 input_channels=3,
                 num_classes=10,
                 initial_channels=32,  # Reduced initial channels
                 num_blocks=[2, 2, 2, 2],
                 hebbian_lr=0.005,  # Reduced hebbian learning rate
                 sparsity_ratio=0.3,  # Reduced sparsity for more connections
                 max_grad_norm=0.5):  # Reduced gradient norm for stability
        super().__init__()
        
        self.initial_channels = initial_channels
        self.max_grad_norm = max_grad_norm
        
        # Initial convolution with smaller channel count
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, initial_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU()
        )
        
        # Build network stages
        self.stages = nn.ModuleList()
        current_channels = initial_channels
        
        # Fixed channel progression
        channels_per_stage = [
            initial_channels,      # Stage 1: 32
            initial_channels * 2,  # Stage 2: 64
            initial_channels * 4,  # Stage 3: 128
            initial_channels * 8   # Stage 4: 256
        ]
        
        # Build stages with fixed channel counts
        for i, (num_blocks_in_stage, out_channels) in enumerate(zip(num_blocks, channels_per_stage)):
            stage = []
            for j in range(num_blocks_in_stage):
                # Only use stride in first block of each stage (except first)
                stride = 2 if j == 0 and i > 0 else 1
                
                block = NHSNetBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    hebbian_lr=hebbian_lr,
                    sparsity_ratio=sparsity_ratio,
                    stride=stride,
                    residual_scale=0.2  # Increased residual scaling
                )
                stage.append(block)
                current_channels = out_channels
                
            self.stages.append(nn.Sequential(*stage))
        
        # Single neurogenesis module for the entire network
        self.neurogenesis = DynamicNeurogenesisModule(
            initial_neurons=channels_per_stage[-1],  # Start from last stage's channels
            max_neurons=channels_per_stage[-1] * 2,  # Maximum double the final channels
            activation_threshold=0.3,  # Increased threshold for more selective growth
            growth_factor=0.03,  # Reduced growth rate for stability
            weight_scale=0.005  # Smaller weight scale for new neurons
        )
        
        # Global pooling and classification
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels_per_stage[-1], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),  # Reduced dropout
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
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def _update_channels(self, block, new_channels):
        """Helper method to update block channels"""
        return block._update_channels(new_channels)
    
    def forward(self, x):
        # Apply gradient clipping to prevent exploding gradients
        if self.training and x.requires_grad:
            x.register_hook(lambda grad: torch.nn.utils.clip_grad_norm_(grad, self.max_grad_norm))
            
        # Get device from input tensor
        device = x.device
        
        # Ensure stem is on the correct device
        self.stem = self.stem.to(device)
        x = self.stem(x)
        
        # Process through stages
        for stage_idx, stage in enumerate(self.stages):
            # Ensure stage is on the correct device
            stage = stage.to(device)
            self.stages[stage_idx] = stage
            
            # Forward through each block in stage
            for block_idx, block in enumerate(stage):
                # Ensure block is on the correct device
                block = block.to(device)
                
                identity = x
                
                # Check for channel mismatch before processing
                if x.size(1) != block.in_channels:
                    # Fix channel mismatch by updating the block
                    block.in_channels = x.size(1)
                    block._make_shortcut()
                
                # Main path
                out = block.conv1(x)
                out = block.bn1(out)
                out = block._apply_layer_norm(out)
                out = F.relu(out)
                
                out = block.sparse_conv(out)
                out = block.bn2(out)
                out = block._apply_layer_norm(out)
                out = block.gating(out)
                
                # Residual connection
                shortcut_out = block.shortcut(identity)
                
                # Residual addition with scaling
                out = out * block.residual_scale + shortcut_out
                out = F.relu(out)
                out = block.dropout(out)
                
                # Handle NaN values
                if torch.isnan(out).any():
                    out = identity
                
                x = out
            
            # Apply neurogenesis only at the end of each stage during training
            # and only with a certain probability to reduce instability
            if self.training and stage_idx < len(self.stages) - 1 and torch.rand(1).item() < 0.3:
                with torch.no_grad():
                    # Ensure neurogenesis module is on the correct device
                    self.neurogenesis = self.neurogenesis.to(device)
                    
                    mean_activation, under_activated = self.neurogenesis.compute_activation_statistics(x)
                    if under_activated.any() and not torch.isnan(mean_activation).any():
                        next_stage = self.stages[stage_idx + 1]
                        # Ensure next stage is on the correct device
                        next_stage = next_stage.to(device)
                        self.stages[stage_idx + 1] = next_stage
                        
                        first_block = next_stage[0]
                        # Ensure first block is on the correct device
                        first_block = first_block.to(device)
                        
                        # Update the first block's input channels to match current output
                        first_block.in_channels = x.size(1)
                        
                        # Try to expand the first block of next stage
                        new_conv1 = self.neurogenesis.expand_layer(first_block.conv1, x)
                        if new_conv1 is not first_block.conv1:
                            new_channels = new_conv1.out_channels
                            if new_channels <= self.neurogenesis.max_neurons:
                                first_block.conv1 = new_conv1.to(device)
                                
                                # Update the block's channels
                                first_block._update_channels(new_channels)
                                
                                # Update all blocks in the next stage
                                for i in range(1, len(next_stage)):
                                    next_block = next_stage[i].to(device)
                                    next_block.in_channels = new_channels
                                    next_block._make_shortcut()
                                    next_stage[i] = next_block
        
        # Ensure final layers are on the correct device
        self.avgpool = self.avgpool.to(device)
        self.classifier = self.classifier.to(device)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
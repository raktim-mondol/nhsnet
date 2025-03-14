import torch
import torch.nn as nn
import torch.nn.functional as F
from .nhsnet import NHSNetBlock, calculate_groups_for_channels, SEBlock
from ..layers.hebbian_conv import HebbianConv2d
from ..layers.structured_sparse import StructuredSparseConv2d
from ..layers.dynamic_neurogenesis import DynamicNeurogenesisModule

class CNNBlock(nn.Module):
    """Traditional CNN block with residual connections and normalization"""
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_se = use_se
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Squeeze-and-Excitation attention
        if use_se:
            self.se = SEBlock(out_channels, reduction=8)
        
        # Shortcut connection
        self._make_shortcut()
        
        # Layer normalization for better stability
        num_groups = calculate_groups_for_channels(out_channels)
        self.layer_norm = nn.GroupNorm(num_groups, out_channels)
        
    def _make_shortcut(self):
        """Create or update shortcut connection"""
        if self.stride != 1 or self.in_channels != self.out_channels:
            # Get device from existing parameters if available
            device = next(self.parameters(), torch.device('cpu')).device
            
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    self.out_channels,
                    kernel_size=1,
                    stride=self.stride,
                    bias=False
                ).to(device),
                nn.BatchNorm2d(self.out_channels).to(device)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        # Get device from input tensor
        device = x.device
        
        # Ensure all components are on the same device
        if next(self.parameters()).device != device:
            self.to(device)
            
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.layer_norm(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.layer_norm(out)
        
        # Apply SE attention if enabled
        if self.use_se:
            out = self.se(out)
        
        # Residual connection
        out = out + self.shortcut(identity)
        out = F.relu(out, inplace=True)
        
        # Handle NaN values
        if torch.isnan(out).any():
            out = identity
            
        return out

class HybridBlock(nn.Module):
    """Hybrid block that combines CNN and NHSNet features"""
    def __init__(self, in_channels, out_channels, stride=1, 
                 hebbian_lr=0.01, sparsity_ratio=0.5, use_se=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # CNN path
        self.cnn_block = CNNBlock(in_channels, out_channels, stride, use_se)
        
        # NHSNet path
        self.nhs_block = NHSNetBlock(
            in_channels, 
            out_channels, 
            hebbian_lr=hebbian_lr,
            sparsity_ratio=sparsity_ratio,
            stride=stride,
            use_se=use_se
        )
        
        # Fusion layer
        self.fusion = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)
        self.fusion_bn = nn.BatchNorm2d(out_channels)
        
        # Adaptive weights for path importance
        self.path_weights = nn.Parameter(torch.ones(2) / 2)
        
    def forward(self, x):
        # Get device from input tensor
        device = x.device
        
        # Ensure all components are on the same device
        if next(self.parameters()).device != device:
            self.to(device)
            
        # Process through both paths
        cnn_out = self.cnn_block(x)
        nhs_out = self.nhs_block(x)
        
        # Normalize path weights
        weights = F.softmax(self.path_weights, dim=0)
        
        # Weighted combination
        out = weights[0] * cnn_out + weights[1] * nhs_out
        
        return out

class CNNNHSNet(nn.Module):
    """Hybrid CNN-NHSNet architecture that combines traditional CNN with neuroplasticity"""
    def __init__(self,
                 input_channels=3,
                 num_classes=10,
                 initial_channels=64,
                 num_blocks=[3, 4, 6, 3],
                 hebbian_lr=0.005,
                 sparsity_ratio=0.3,
                 max_grad_norm=0.5,
                 dropout_rate=0.2,
                 use_se=True,
                 hybrid_mode='parallel'):
        super().__init__()
        
        self.initial_channels = initial_channels
        self.max_grad_norm = max_grad_norm
        self.hybrid_mode = hybrid_mode  # 'parallel', 'sequential', or 'adaptive'
        
        # Improved stem with more channels and better initialization
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, initial_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_channels, initial_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True)
        )
        
        # Build network stages
        self.stages = nn.ModuleList()
        current_channels = initial_channels
        
        # Fixed channel progression with wider networks
        channels_per_stage = [
            initial_channels,      # Stage 1
            initial_channels * 2,  # Stage 2
            initial_channels * 4,  # Stage 3
            initial_channels * 8   # Stage 4
        ]
        
        # Build stages with fixed channel counts
        for i, (num_blocks_in_stage, out_channels) in enumerate(zip(num_blocks, channels_per_stage)):
            stage = []
            for j in range(num_blocks_in_stage):
                # Only use stride in first block of each stage (except first)
                stride = 2 if j == 0 and i > 0 else 1
                
                if hybrid_mode == 'parallel':
                    # Use hybrid blocks that combine CNN and NHSNet
                    block = HybridBlock(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        stride=stride,
                        hebbian_lr=hebbian_lr,
                        sparsity_ratio=sparsity_ratio,
                        use_se=use_se
                    )
                elif hybrid_mode == 'sequential':
                    # Alternate between CNN and NHSNet blocks
                    if j % 2 == 0:
                        block = CNNBlock(
                            in_channels=current_channels,
                            out_channels=out_channels,
                            stride=stride,
                            use_se=use_se
                        )
                    else:
                        block = NHSNetBlock(
                            in_channels=current_channels,
                            out_channels=out_channels,
                            hebbian_lr=hebbian_lr,
                            sparsity_ratio=sparsity_ratio,
                            stride=stride,
                            use_se=use_se,
                            dropout_rate=dropout_rate
                        )
                else:  # adaptive
                    # Use CNN blocks in early stages, NHSNet in later stages
                    if i < 2:
                        block = CNNBlock(
                            in_channels=current_channels,
                            out_channels=out_channels,
                            stride=stride,
                            use_se=use_se
                        )
                    else:
                        block = NHSNetBlock(
                            in_channels=current_channels,
                            out_channels=out_channels,
                            hebbian_lr=hebbian_lr,
                            sparsity_ratio=sparsity_ratio,
                            stride=stride,
                            use_se=use_se,
                            dropout_rate=dropout_rate
                        )
                
                stage.append(block)
                current_channels = out_channels
                
            self.stages.append(nn.Sequential(*stage))
        
        # Neurogenesis module for the entire network
        self.neurogenesis = DynamicNeurogenesisModule(
            initial_neurons=channels_per_stage[-1],
            max_neurons=channels_per_stage[-1] * 2,
            activation_threshold=0.3,
            growth_factor=0.03,
            weight_scale=0.005
        )
        
        # Global pooling and improved classification head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels_per_stage[-1], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate/2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Mixup augmentation strength
        self.mixup_alpha = 0.2
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, HebbianConv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _mixup(self, x, targets):
        """Apply mixup augmentation during training"""
        if self.training and self.mixup_alpha > 0 and targets is not None:
            batch_size = x.size(0)
            lam = torch.distributions.beta.Beta(self.mixup_alpha, self.mixup_alpha).sample().to(x.device)
            index = torch.randperm(batch_size).to(x.device)
            
            mixed_x = lam * x + (1 - lam) * x[index]
            targets_a, targets_b = targets, targets[index]
            return mixed_x, targets_a, targets_b, lam
        return x, targets, None, None
    
    def forward(self, x, targets=None):
        # Apply mixup if targets are provided
        lam = None
        targets_a = targets_b = None
        if targets is not None:
            x, targets_a, targets_b, lam = self._mixup(x, targets)
        
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
            
            # Forward through the stage
            x = stage(x)
            
            # Apply neurogenesis only at the end of each stage during training
            # and only with a certain probability to reduce instability
            if self.training and stage_idx < len(self.stages) - 1 and torch.rand(1).item() < 0.3:
                with torch.no_grad():
                    # Ensure neurogenesis module is on the correct device
                    self.neurogenesis = self.neurogenesis.to(device)
                    
                    # Get the next stage and its first block
                    next_stage = self.stages[stage_idx + 1]
                    next_stage = next_stage.to(device)
                    self.stages[stage_idx + 1] = next_stage
                    
                    first_block = next_stage[0]
                    first_block = first_block.to(device)
                    
                    # Skip neurogenesis if there's already a channel mismatch
                    if x.size(1) != first_block.in_channels:
                        # Just update the block's input channels
                        first_block.in_channels = x.size(1)
                        if hasattr(first_block, '_make_shortcut'):
                            first_block._make_shortcut()
                        continue
                    
                    # Compute activation statistics
                    mean_activation, under_activated = self.neurogenesis.compute_activation_statistics(x)
                    
                    if under_activated.any() and not torch.isnan(mean_activation).any():
                        # Update the first block's input channels to match current output
                        first_block.in_channels = x.size(1)
                        
                        # Try to expand the first block
                        if hasattr(first_block, 'conv1'):
                            new_conv1 = self.neurogenesis.expand_layer(first_block.conv1, x)
                            if new_conv1 is not first_block.conv1:
                                new_channels = new_conv1.out_channels
                                if new_channels <= self.neurogenesis.max_neurons:
                                    first_block.conv1 = new_conv1.to(device)
                                    
                                    # Update the block's channels
                                    if hasattr(first_block, '_update_channels'):
                                        first_block._update_channels(new_channels)
                                    
                                    # Update all blocks in the next stage
                                    for i in range(1, len(next_stage)):
                                        next_block = next_stage[i].to(device)
                                        next_block.in_channels = new_channels
                                        if hasattr(next_block, '_make_shortcut'):
                                            next_block._make_shortcut()
                                        next_stage[i] = next_block
        
        # Ensure final layers are on the correct device
        self.avgpool = self.avgpool.to(device)
        self.classifier = self.classifier.to(device)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        # Apply mixup loss if used
        if self.training and lam is not None and targets_a is not None and targets_b is not None:
            return x, targets_a, targets_b, lam
        
        return x 
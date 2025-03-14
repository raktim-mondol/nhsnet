import torch
import torch.nn as nn
import numpy as np

class StructuredSparseConv2d(nn.Conv2d):
    """Convolutional layer with structured sparsity pattern"""
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 sparsity_ratio=0.5, pattern='hexagonal', weight_scale=0.01, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.sparsity_ratio = sparsity_ratio
        self.pattern = pattern
        self.weight_scale = weight_scale
        self.register_buffer('mask', self._create_sparse_mask())
        
        # Apply mask to weights
        self.weight.data.mul_(self.mask)
        
        # Initialize weights properly
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
            
        # Scale weights to prevent explosion
        with torch.no_grad():
            weight_norm = torch.norm(self.weight.data.view(self.out_channels, -1), dim=1, keepdim=True)
            weight_norm = weight_norm.view(self.out_channels, 1, 1, 1)
            self.weight.data = self.weight.data / (weight_norm + 1e-6) * self.weight_scale
        
    def _create_sparse_mask(self):
        if self.pattern == 'hexagonal':
            return self._create_hexagonal_mask()
        else:
            return self._create_random_mask()
            
    def _create_hexagonal_mask(self):
        mask = torch.ones_like(self.weight)
        k = self.kernel_size[0]
        center = k // 2
        
        for i in range(k):
            for j in range(k):
                if (i + j) % 2 == 0:  # Hexagonal pattern
                    mask[:, :, i, j] = 0
                    
        return mask
        
    def _create_random_mask(self):
        """Create random sparse mask"""
        mask = torch.ones_like(self.weight)
        n_prune = int(self.sparsity_ratio * mask.numel())
        
        # Generate random indices to prune
        flat_indices = torch.randperm(mask.numel())[:n_prune]
        flat_mask = torch.ones(mask.numel(), device=mask.device)
        flat_mask[flat_indices] = 0
        
        # Reshape back to original shape
        return flat_mask.view_as(mask)
        
    def forward(self, x):
        # Get device from input tensor
        device = x.device
        
        # Ensure weights and mask are on the same device as input
        if self.weight.device != device:
            self.weight = self.weight.to(device)
            self.mask = self.mask.to(device)
            if self.bias is not None:
                self.bias = self.bias.to(device)
        
        # Check for input channel mismatch and fix if needed
        if x.size(1) != self.in_channels:
            # Create a new layer with correct input channels
            new_layer = StructuredSparseConv2d(
                x.size(1), self.out_channels, self.kernel_size[0],
                sparsity_ratio=self.sparsity_ratio, pattern=self.pattern,
                weight_scale=self.weight_scale,
                stride=self.stride, padding=self.padding, 
                dilation=self.dilation, groups=self.groups,
                bias=self.bias is not None
            ).to(device)
            # Transfer existing weights where possible
            with torch.no_grad():
                min_channels = min(x.size(1), self.in_channels)
                new_layer.weight.data[:, :min_channels].copy_(
                    self.weight.data[:, :min_channels]
                )
                if self.bias is not None:
                    new_layer.bias.data.copy_(self.bias.data)
            
            # Replace self with new layer
            self.in_channels = new_layer.in_channels
            self.weight = new_layer.weight
            self.mask = new_layer.mask
            if self.bias is not None:
                self.bias = new_layer.bias
        
        # Ensure weights remain sparse
        self.weight.data.mul_(self.mask)
        
        # Check for NaN values in weights and fix
        if torch.isnan(self.weight.data).any():
            with torch.no_grad():
                nan_mask = torch.isnan(self.weight.data)
                self.weight.data[nan_mask] = 0.0
                
        # Normalize weights periodically to prevent explosion
        if self.training and torch.rand(1).item() < 0.1:  # 10% chance each forward pass
            with torch.no_grad():
                weight_norm = torch.norm(self.weight.data.view(self.out_channels, -1), dim=1, keepdim=True)
                weight_norm = weight_norm.view(self.out_channels, 1, 1, 1)
                self.weight.data = self.weight.data / (weight_norm + 1e-6) * self.weight_scale
        
        return super().forward(x)